"""Search across multi-representation embeddings.

Given a query embedding, search `knowledge_representations` table (migration 002)
separately for each representation type (summary/keywords/questions — raw is
already covered by the main `embeddings` table) and fuse the per-representation
ranked lists via RRF.

Returns (knowledge_id, fused_score) pairs. If the table is empty or no matches
found, returns an empty list (safe no-op tier).
"""

from __future__ import annotations

import sqlite3
import struct
import sys
from typing import Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from multi_repr_store import rrf_fuse
except ImportError:  # package path
    from .multi_repr_store import rrf_fuse  # type: ignore[no-redef]

LOG = lambda msg: sys.stderr.write(f"[multi-repr-search] {msg}\n")


# LLM-generated views we search over (raw already covered by embeddings table)
_SEARCH_REPRESENTATIONS: tuple[str, ...] = (
    "summary", "keywords", "questions", "compressed",
)


def _cosine(a: list[float], b: list[float]) -> float:
    if np is None:
        num = sum(x * y for x, y in zip(a, b))
        da = sum(x * x for x in a) ** 0.5
        db = sum(y * y for y in b) ** 0.5
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))


def _unpack_vector(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


def has_representations(db: sqlite3.Connection) -> bool:
    """Cheap existence check to gate this tier in hot path."""
    try:
        row = db.execute(
            "SELECT 1 FROM knowledge_representations LIMIT 1"
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def search(
    db: sqlite3.Connection,
    query_embedding: list[float],
    project: str | None = None,
    n_candidates: int = 100,
    top_n: int = 20,
) -> list[tuple[int, float]]:
    """Search each representation, fuse with RRF, return (knowledge_id, score).

    Scores returned are RRF fusion scores (not cosine similarities) — use as
    one tier among many in the caller's fusion.
    """
    if not query_embedding:
        return []

    per_repr: dict[str, list[tuple[int, float]]] = {}

    for repr_name in _SEARCH_REPRESENTATIONS:
        try:
            rows = _fetch(db, repr_name, project, n_candidates)
        except sqlite3.Error as e:
            LOG(f"fetch error for representation={repr_name}: {e}")
            continue
        if not rows:
            continue

        scored: list[tuple[int, float]] = []
        for r in rows:
            try:
                vec = _unpack_vector(r["float32_vector"], r["embed_dim"])
            except (struct.error, KeyError):
                continue
            if len(vec) != len(query_embedding):
                # Dim mismatch — different embedder. Skip silently.
                continue
            sim = _cosine(query_embedding, vec)
            if sim > 0:
                scored.append((r["knowledge_id"], sim))

        if scored:
            scored.sort(key=lambda kv: kv[1], reverse=True)
            per_repr[repr_name] = scored[:top_n]

    if not per_repr:
        return []

    fused = rrf_fuse(per_repr, k=60, top_n=top_n)
    return fused


def _fetch(
    db: sqlite3.Connection,
    representation: str,
    project: str | None,
    limit: int,
) -> list[sqlite3.Row]:
    if project:
        return db.execute(
            """SELECT kr.knowledge_id, kr.float32_vector, kr.embed_dim
                 FROM knowledge_representations kr
                 JOIN knowledge k ON k.id = kr.knowledge_id
                WHERE kr.representation = ?
                  AND k.status = 'active'
                  AND k.project = ?
                LIMIT ?""",
            (representation, project, limit),
        ).fetchall()
    return db.execute(
        """SELECT kr.knowledge_id, kr.float32_vector, kr.embed_dim
             FROM knowledge_representations kr
             JOIN knowledge k ON k.id = kr.knowledge_id
            WHERE kr.representation = ?
              AND k.status = 'active'
            LIMIT ?""",
        (representation, limit),
    ).fetchall()
