"""v11.0 Phase W1-F — Global (project-wide) Entity Resolver.

The session-local `coref_resolver` rewrites pronouns inside a single save
using recent records from the same session. It cannot tell the retriever
that "Sarah", "Dr. Sarah Williams" and yesterday's "she" all designate
the same person. This module fills that gap:

  resolve(conn, mention, project, type_, embed_fn=...) -> ResolveResult

Lookup order, cheapest first:

    1. Pronoun guard. Pronouns ("she", "they", "она", …) never become
       canonicals — caller must resolve antecedent itself. We return a
       sentinel ResolveResult(canonical_id=-1, matched_via="pronoun")
       so the caller can branch without try/except.

    2. Exact match on `canonical_entities.name_norm`.
       O(1) lookup via UNIQUE(project, type, name_norm).

    3. Alias match on `entity_aliases.alias_norm` joined to
       canonical_entities filtered by (project, type).

    4. Embedding match. Cosine similarity vs every canonical of the
       same (project, type) pair. If max ≥ threshold → reuse and
       insert the surface form as an alias (source="embedding_match").

    5. Insert as new canonical (if `create_if_missing`) and seed an
       alias row with the canonical name itself, so the next exact-form
       call finds it via step 3 too.

Normalization is unicode-aware: we lower-case, NFKD-decompose, strip
combining marks ("Sara Williams" == "Sára Williams" == "sara williams"),
collapse whitespace and remove punctuation. Empty/whitespace-only
mentions raise `ValueError` — callers that might pass such strings
should pre-filter.

The resolver never touches the network: caller injects `embed_fn`. In
production this is bound to `EmbeddingProvider.embed_query`; tests pass
a deterministic hash-based stub.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Pronouns and discourse deictics that must never become canonical
# entities. Multilingual; lower-cased; matched against the *normalized*
# mention so "She" / "ОНА" / "Они" all hit. Subset of the surface forms
# scanned by `coref_resolver.needs_resolution` — only those that, by
# themselves, would otherwise be inserted as a canonical name.
_PRONOUNS: frozenset[str] = frozenset({
    # English personal pronouns
    "he", "him", "his",
    "she", "her", "hers",
    "they", "them", "their", "theirs",
    "it", "its",
    "we", "us", "our", "ours",
    "i", "me", "my", "mine",
    "you", "your", "yours",
    # English deictics that look entity-like
    "this", "that", "these", "those",
    "here", "there",
    # Russian personal pronouns and deictics
    "он", "она", "оно", "они",
    "его", "её", "ее", "их",
    "это", "этот", "эта", "эти",
    "то", "та", "те", "тот",
    "мы", "нас", "наш", "наша", "наши",
    "я", "меня", "мой", "моя", "мои",
    "ты", "тебя", "твой", "твоя", "твои",
    "вы", "вас", "ваш", "ваша", "ваши",
})

# Whitespace runs collapse to single space. Anything in the unicode "P"
# (punctuation) or "S" (symbol) categories is stripped after NFKD —
# this also kills periods inside "Dr." and hyphens in "Sara-Jane".
_WS_RE = re.compile(r"\s+", re.UNICODE)


# ──────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────


@dataclass
class ResolveResult:
    """Outcome of a resolve() call.

    * `canonical_id == -1` AND `matched_via == "pronoun"` — caller passed
      a bare pronoun; nothing was inserted, antecedent resolution is
      caller's job.
    * `is_new == True` — a brand-new canonical was created on this call.
      The caller may want to log/notify or attach attrs.
    * `confidence` is 1.0 for exact/alias/created, the cosine value for
      embedding matches.
    """

    canonical_id: int
    canonical_name: str
    matched_via: str          # "exact" | "alias" | "embedding" | "created" | "pronoun"
    confidence: float
    is_new: bool


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────


def normalize(mention: str) -> str:
    """Canonical surface-form key for lookup.

    NFKD + drop combining marks (accents) → casefold → strip
    punctuation/symbols → collapse whitespace. Non-empty input that
    consists entirely of punctuation collapses to empty string and the
    caller should treat that as "unresolvable".
    """
    if mention is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", mention)
    out_chars: list[str] = []
    for ch in decomposed:
        cat = unicodedata.category(ch)
        # Mn = combining mark (e.g. accent over a letter) — drop
        if cat == "Mn":
            continue
        # P* = punctuation, S* = symbol — replace with space so token
        # boundaries survive ("dr.sarah" → "dr sarah", not "drsarah")
        if cat.startswith("P") or cat.startswith("S"):
            out_chars.append(" ")
            continue
        out_chars.append(ch)
    folded = "".join(out_chars).casefold().strip()
    return _WS_RE.sub(" ", folded)


def is_pronoun(mention: str) -> bool:
    """True if the *normalized* mention is a known pronoun/deictic."""
    return normalize(mention) in _PRONOUNS


# ──────────────────────────────────────────────
# Embedding (de)serialization
# ──────────────────────────────────────────────


def _vec_to_blob(vec: Iterable[float]) -> bytes:
    arr = np.asarray(list(vec), dtype=np.float32)
    return struct.pack(f"{arr.size}f", *arr.tolist())


def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    n = len(blob) // 4
    if n == 0:
        return None
    return np.asarray(struct.unpack(f"{n}f", blob), dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ──────────────────────────────────────────────
# Internal lookup helpers
# ──────────────────────────────────────────────


def _exact_lookup(
    conn: sqlite3.Connection,
    *,
    project: str,
    type_: str,
    name_norm: str,
) -> tuple[int, str] | None:
    row = conn.execute(
        """SELECT id, name FROM canonical_entities
           WHERE project = ? AND type = ? AND name_norm = ?
           LIMIT 1""",
        (project, type_, name_norm),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), str(row[1])


def _alias_lookup(
    conn: sqlite3.Connection,
    *,
    project: str,
    type_: str,
    alias_norm: str,
) -> tuple[int, str] | None:
    row = conn.execute(
        """SELECT c.id, c.name
           FROM entity_aliases a
           JOIN canonical_entities c ON c.id = a.canonical_id
           WHERE a.alias_norm = ? AND c.project = ? AND c.type = ?
           ORDER BY a.confidence DESC, a.id ASC
           LIMIT 1""",
        (alias_norm, project, type_),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), str(row[1])


def _alias_exists(
    conn: sqlite3.Connection,
    *,
    canonical_id: int,
    alias_norm: str,
) -> bool:
    row = conn.execute(
        """SELECT 1 FROM entity_aliases
           WHERE canonical_id = ? AND alias_norm = ?
           LIMIT 1""",
        (canonical_id, alias_norm),
    ).fetchone()
    return row is not None


def _embedding_match(
    conn: sqlite3.Connection,
    *,
    project: str,
    type_: str,
    query_vec: np.ndarray,
    threshold: float,
) -> tuple[int, str, float] | None:
    """Linear scan of (project, type) canonicals — fine up to a few
    thousand entities per (project, type) cell. If it ever bites, swap
    the BLOB column for sqlite-vec or chroma."""
    rows = conn.execute(
        """SELECT id, name, embedding FROM canonical_entities
           WHERE project = ? AND type = ?""",
        (project, type_),
    ).fetchall()
    best_id: int | None = None
    best_name: str = ""
    best_sim: float = -1.0
    for row in rows:
        cand_id = int(row[0])
        cand_name = str(row[1])
        vec = _blob_to_vec(row[2])
        if vec is None or vec.size != query_vec.size:
            continue
        sim = _cosine(query_vec, vec)
        if sim > best_sim:
            best_sim = sim
            best_id = cand_id
            best_name = cand_name
    if best_id is None or best_sim < threshold:
        return None
    return best_id, best_name, best_sim


def _safe_embed(embed_fn: Callable[[str], np.ndarray] | None, text: str) -> np.ndarray | None:
    """Wrap caller-supplied embed_fn so resolver degrades gracefully.

    If embedding is unavailable (no fn, fn raises, fn returns empty
    vector) we just skip the embedding-match step. The resolver still
    works on exact/alias/create paths.
    """
    if embed_fn is None:
        return None
    try:
        raw = embed_fn(text)
    except Exception:  # noqa: BLE001 — embedding is best-effort
        return None
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return arr


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def resolve(
    conn: sqlite3.Connection,
    mention: str,
    project: str,
    type_: str,
    *,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    threshold: float = 0.85,
    create_if_missing: bool = True,
) -> ResolveResult:
    """Resolve a surface mention to a canonical entity id.

    See module docstring for the full lookup order. Pronouns short-circuit
    to a sentinel result (canonical_id=-1). All other inputs are validated:
    empty / whitespace-only / punctuation-only mentions raise ValueError
    so callers don't silently insert garbage canonicals.
    """
    if not isinstance(mention, str):
        raise TypeError(f"mention must be str, got {type(mention).__name__}")
    if not mention.strip():
        raise ValueError("mention must be non-empty")
    if not project:
        raise ValueError("project must be non-empty")
    if not type_:
        raise ValueError("type_ must be non-empty")

    norm = normalize(mention)
    if not norm:
        raise ValueError(
            f"mention {mention!r} normalizes to empty string "
            "(consists only of punctuation/whitespace)"
        )

    # 1. Pronoun guard
    if norm in _PRONOUNS:
        return ResolveResult(
            canonical_id=-1,
            canonical_name=mention,
            matched_via="pronoun",
            confidence=0.0,
            is_new=False,
        )

    # 2. Exact match on canonical name
    hit = _exact_lookup(conn, project=project, type_=type_, name_norm=norm)
    if hit is not None:
        cid, cname = hit
        return ResolveResult(
            canonical_id=cid,
            canonical_name=cname,
            matched_via="exact",
            confidence=1.0,
            is_new=False,
        )

    # 3. Alias match
    hit = _alias_lookup(conn, project=project, type_=type_, alias_norm=norm)
    if hit is not None:
        cid, cname = hit
        return ResolveResult(
            canonical_id=cid,
            canonical_name=cname,
            matched_via="alias",
            confidence=1.0,
            is_new=False,
        )

    # 4. Embedding match
    query_vec = _safe_embed(embed_fn, mention)
    if query_vec is not None:
        emb_hit = _embedding_match(
            conn,
            project=project,
            type_=type_,
            query_vec=query_vec,
            threshold=threshold,
        )
        if emb_hit is not None:
            cid, cname, sim = emb_hit
            # Promote the surface form to a stored alias so future
            # lookups skip the linear scan.
            add_alias(
                conn,
                canonical_id=cid,
                alias=mention,
                source="embedding_match",
                confidence=sim,
            )
            return ResolveResult(
                canonical_id=cid,
                canonical_name=cname,
                matched_via="embedding",
                confidence=sim,
                is_new=False,
            )

    # 5. Create new canonical
    if not create_if_missing:
        # Caller wants read-only resolution; surface the miss explicitly
        # via canonical_id=0 so the type stays int (not Optional[int]).
        return ResolveResult(
            canonical_id=0,
            canonical_name=mention,
            matched_via="miss",
            confidence=0.0,
            is_new=False,
        )

    canonical_blob = _vec_to_blob(query_vec) if query_vec is not None else None

    cur = conn.execute(
        """INSERT INTO canonical_entities (project, type, name, name_norm, embedding, attrs)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (project, type_, mention.strip(), norm, canonical_blob, None),
    )
    new_id = int(cur.lastrowid)

    # Seed alias row for the canonical name itself so a future call with
    # the exact same surface form takes the alias-lookup fast path too.
    add_alias(
        conn,
        canonical_id=new_id,
        alias=mention.strip(),
        source="explicit",
        confidence=1.0,
    )
    conn.commit()

    return ResolveResult(
        canonical_id=new_id,
        canonical_name=mention.strip(),
        matched_via="created",
        confidence=1.0,
        is_new=True,
    )


def add_alias(
    conn: sqlite3.Connection,
    canonical_id: int,
    alias: str,
    source: str,
    confidence: float,
) -> None:
    """Attach a surface form to an existing canonical.

    Idempotent: a duplicate (canonical_id, alias_norm) pair is silently
    skipped — repeat saves of the same mention shouldn't bloat the table.
    """
    if canonical_id <= 0:
        raise ValueError(f"canonical_id must be positive, got {canonical_id}")
    if not alias or not alias.strip():
        raise ValueError("alias must be non-empty")
    alias_norm = normalize(alias)
    if not alias_norm:
        raise ValueError(f"alias {alias!r} normalizes to empty string")
    if _alias_exists(conn, canonical_id=canonical_id, alias_norm=alias_norm):
        return
    conn.execute(
        """INSERT INTO entity_aliases
               (canonical_id, alias, alias_norm, source, confidence)
           VALUES (?, ?, ?, ?, ?)""",
        (canonical_id, alias.strip(), alias_norm, source, float(confidence)),
    )
    conn.commit()


def merge_canonicals(
    conn: sqlite3.Connection,
    keep_id: int,
    drop_ids: list[int],
) -> int:
    """Move every alias from `drop_ids` to `keep_id`, then delete the
    drops. Returns total number of alias rows moved.

    The drop canonicals' own names are preserved as aliases on the
    surviving canonical (with source="merge") so retrieval that
    previously matched a drop's exact name still resolves.
    """
    if keep_id <= 0:
        raise ValueError(f"keep_id must be positive, got {keep_id}")
    drops = [int(x) for x in drop_ids if int(x) != keep_id and int(x) > 0]
    if not drops:
        return 0

    # Sanity: keep_id must exist.
    keep_row = conn.execute(
        "SELECT id FROM canonical_entities WHERE id = ?", (keep_id,)
    ).fetchone()
    if keep_row is None:
        raise ValueError(f"keep_id {keep_id} does not exist")

    moved = 0
    placeholders = ",".join("?" * len(drops))

    # First, fold each drop's canonical name into the surviving alias set
    # (skipping any that would collide with an existing alias_norm on
    # keep_id — _alias_exists handles dedup).
    drop_rows = conn.execute(
        f"SELECT id, name FROM canonical_entities WHERE id IN ({placeholders})",
        drops,
    ).fetchall()
    for row in drop_rows:
        try:
            add_alias(
                conn,
                canonical_id=keep_id,
                alias=str(row[1]),
                source="merge",
                confidence=1.0,
            )
        except ValueError:
            # Defensive: drop canonical with weird name — skip rather
            # than abort the whole merge.
            continue

    # Move every alias row, deduping against the surviving canonical's
    # existing alias_norms.
    alias_rows = conn.execute(
        f"""SELECT id, alias_norm FROM entity_aliases
            WHERE canonical_id IN ({placeholders})""",
        drops,
    ).fetchall()
    for arow in alias_rows:
        alias_id = int(arow[0])
        alias_norm = str(arow[1])
        if _alias_exists(conn, canonical_id=keep_id, alias_norm=alias_norm):
            # Drop the redundant alias along with its old canonical.
            conn.execute("DELETE FROM entity_aliases WHERE id = ?", (alias_id,))
            continue
        conn.execute(
            "UPDATE entity_aliases SET canonical_id = ? WHERE id = ?",
            (keep_id, alias_id),
        )
        moved += 1

    # Finally, delete the dropped canonicals. ON DELETE CASCADE on
    # entity_aliases.canonical_id sweeps any leftover alias rows that
    # weren't migrated above (shouldn't happen — defence in depth).
    conn.execute(
        f"DELETE FROM canonical_entities WHERE id IN ({placeholders})",
        drops,
    )
    conn.commit()
    return moved


def list_aliases(conn: sqlite3.Connection, canonical_id: int) -> list[str]:
    """All surface forms registered for a canonical, oldest first."""
    if canonical_id <= 0:
        return []
    rows = conn.execute(
        """SELECT alias FROM entity_aliases
           WHERE canonical_id = ?
           ORDER BY id ASC""",
        (canonical_id,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def get_canonical(
    conn: sqlite3.Connection,
    canonical_id: int,
) -> dict | None:
    """Read-only fetch by id. Convenience for callers/tests."""
    if canonical_id <= 0:
        return None
    row = conn.execute(
        """SELECT id, project, type, name, name_norm, attrs, created_at
           FROM canonical_entities WHERE id = ?""",
        (canonical_id,),
    ).fetchone()
    if row is None:
        return None
    raw_attrs = row[5]
    try:
        attrs = json.loads(raw_attrs) if raw_attrs else None
    except (TypeError, ValueError):
        attrs = None
    return {
        "id": int(row[0]),
        "project": str(row[1]),
        "type": str(row[2]),
        "name": str(row[3]),
        "name_norm": str(row[4]),
        "attrs": attrs,
        "created_at": str(row[6]),
    }


__all__ = [
    "ResolveResult",
    "resolve",
    "add_alias",
    "merge_canonicals",
    "list_aliases",
    "get_canonical",
    "normalize",
    "is_pronoun",
]
