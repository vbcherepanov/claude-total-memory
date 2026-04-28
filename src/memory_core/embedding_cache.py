"""v11.0 Phase 7 — Persistent embedding cache (multi-space aware).

Caches the float32 output of the embedder keyed by
`sha256(provider || model || embedding_space || normalized_content)`.

Why a fresh table (`embedding_cache_v11`) instead of the v9 one:
v10's `embedding_cache` (migration 014) keys entries by sha256(text)
only — there is no provider, no model, no embedding-space dimension. In
v11 the same text can produce different vectors per space (text vs code
model) and per provider, so the v10 key would collide silently. The new
table adds those four dimensions to the key + a `last_used_at` column
for LRU eviction.

Public API:
    cache_key(provider, model, space, normalized_content) -> str
    get(db, cache_key) -> list[float] | None      (bumps hit_count + lru ts)
    put(db, cache_key, vector, *, provider, model, space) -> None
    vacuum(db, max_rows=100_000) -> int           (LRU eviction → returns deleted)
    stats(db) -> {"rows": int, "total_hits": int, "by_space": {space: count}}

The vector blob format matches `Store._float32_to_blob`
(struct.pack('f'*dim, *vec)) so the binary-search hot path can reuse a
cached vector with no copy.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from datetime import datetime, timezone
from typing import Iterable, Optional

__all__ = [
    "cache_key",
    "get",
    "put",
    "vacuum",
    "stats",
    "pack_vector",
    "unpack_vector",
]


# ─── Hashing ────────────────────────────────────────────────────────────


def cache_key(
    provider: str,
    model: str,
    space: str,
    normalized_content: str,
) -> str:
    """Deterministic cache key.

    `normalized_content` should already be lowercased / whitespace-collapsed
    (see `memory_core.dedup.normalize`). Mixing in provider/model/space
    ensures the same text never collides across backends or spaces.
    """
    blob = "\x00".join(
        [
            (provider or "").strip().lower(),
            (model or "").strip(),
            (space or "").strip().lower(),
            normalized_content or "",
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ─── Vector blob (matches Store._float32_to_blob) ───────────────────────


def pack_vector(vec: Iterable[float]) -> bytes:
    """Pack a float32 vector into a BLOB.

    Same format as `server.Store._float32_to_blob` — kept inline so this
    module has zero dependency on `server.py` (which would create a
    circular import: server imports memory_core, memory_core would import
    server). Three lines of struct.pack is cheaper than that.
    """
    arr = list(vec)
    return struct.pack(f"{len(arr)}f", *arr)


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of `pack_vector`. Mirrors `Store._blob_to_float32`."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ─── DB operations ──────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get(db: sqlite3.Connection, key: str) -> Optional[list[float]]:
    """Look up a cached vector. Bumps `hit_count` + `last_used_at` on hit.

    Returns None on miss. Raises nothing on a missing table — the v9
    cache and v11 cache may live side by side during the deprecation
    window, so a v9-only DB simply gets a miss.
    """
    if not key:
        return None
    try:
        row = db.execute(
            "SELECT vector_blob FROM embedding_cache_v11 WHERE cache_key=?",
            (key,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    blob = row[0] if not isinstance(row, sqlite3.Row) else row["vector_blob"]
    # Bump LRU timestamp + hit counter. Best-effort: a write failure here
    # MUST NOT shadow the actual cache hit (e.g. read-only replicas).
    try:
        db.execute(
            "UPDATE embedding_cache_v11 "
            "   SET hit_count = hit_count + 1, "
            "       last_used_at = ? "
            " WHERE cache_key = ?",
            (_now_iso(), key),
        )
        db.commit()
    except sqlite3.Error:
        pass
    return unpack_vector(blob)


def put(
    db: sqlite3.Connection,
    key: str,
    vector: Iterable[float],
    *,
    provider: str,
    model: str,
    space: str,
) -> None:
    """Insert-or-replace a cache entry.

    Idempotent: a second put() for the same key resets `last_used_at`
    (recent activity) and `hit_count` to 0 (the row is effectively new).
    """
    if not key:
        return
    vec = list(vector)
    if not vec:
        return
    blob = pack_vector(vec)
    now = _now_iso()
    try:
        db.execute(
            "INSERT OR REPLACE INTO embedding_cache_v11 "
            "(cache_key, provider, model, embedding_space, dim, "
            " vector_blob, hit_count, last_used_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                key,
                (provider or "").strip().lower() or "unknown",
                (model or "").strip() or "unknown",
                (space or "text").strip().lower(),
                len(vec),
                blob,
                now,
                now,
            ),
        )
        db.commit()
    except sqlite3.OperationalError:
        # Table missing (migration 022 not applied) — silent no-op so the
        # caller's hot path never breaks on a fresh DB.
        return


def vacuum(db: sqlite3.Connection, max_rows: int = 100_000) -> int:
    """Evict least-recently-used rows until row count <= `max_rows`.

    Returns the number of deleted rows. Cheap when already under the
    limit (one COUNT, zero DELETE).
    """
    if max_rows < 0:
        max_rows = 0
    try:
        n = db.execute(
            "SELECT COUNT(*) FROM embedding_cache_v11"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return 0
    if n <= max_rows:
        return 0
    surplus = n - max_rows
    cur = db.execute(
        "DELETE FROM embedding_cache_v11 "
        " WHERE cache_key IN ("
        "    SELECT cache_key FROM embedding_cache_v11 "
        "     ORDER BY last_used_at ASC "
        "     LIMIT ?"
        ")",
        (surplus,),
    )
    db.commit()
    return int(cur.rowcount or 0)


def stats(db: sqlite3.Connection) -> dict:
    """Return a small dict suitable for `memory_perf_report`."""
    try:
        rows = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) "
            "FROM embedding_cache_v11"
        ).fetchone()
        by_space_rows = db.execute(
            "SELECT embedding_space, COUNT(*) "
            "FROM embedding_cache_v11 "
            "GROUP BY embedding_space"
        ).fetchall()
    except sqlite3.OperationalError:
        return {"rows": 0, "total_hits": 0, "by_space": {}}
    by_space: dict[str, int] = {}
    for r in by_space_rows:
        space = r[0] if not isinstance(r, sqlite3.Row) else r[0]
        count = r[1] if not isinstance(r, sqlite3.Row) else r[1]
        by_space[str(space or "unknown")] = int(count or 0)
    return {
        "rows": int((rows[0] if rows else 0) or 0),
        "total_hits": int((rows[1] if rows else 0) or 0),
        "by_space": by_space,
    }
