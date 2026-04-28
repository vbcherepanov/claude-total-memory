"""v11.0 Phase 3 — Cache facade.

Re-exports the existing two-level cache (`cache_layer.TwoLevelCache`,
`L1QueryCache`, `L2EmbeddingCache`) plus a tiny LRU helper used by the
embedding hot path. Keep this module dumb — the heavy lifting lives in
`src/cache_layer.py` already.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cache_layer import (  # noqa: E402, F401 — public re-exports
    L1QueryCache,
    L2EmbeddingCache,
    TwoLevelCache,
    make_l1_key,
    make_l2_key,
)


# Public alias matching the legacy name used by some call sites.
CacheLayer = TwoLevelCache


# ─── tiny LRU ─────────────────────────────────────────────────────────


class LRU:
    """Trivial thread-safe LRU. For local use inside hot loops where the
    full L1/L2 stack is overkill (e.g. per-process memoization)."""

    __slots__ = ("_max", "_data", "_lock")

    def __init__(self, max_size: int = 256) -> None:
        self._max = max(1, int(max_size))
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def embedding_cache_key(
    provider: str,
    model: str,
    space: str,
    normalized_content: str,
) -> str:
    """Deterministic key for embedding caches.

    `normalized_content` should already be lowercased/whitespace-collapsed
    (see :func:`memory_core.dedup.normalize`). Mixing in provider/model/
    space ensures the same text never collides across backends.
    """
    blob = "".join(
        [
            (provider or "").strip().lower(),
            (model or "").strip(),
            (space or "").strip().lower(),
            normalized_content or "",
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ─── v11 Phase 7 — persistent embedding cache (multi-space) ──────────
# Thin re-exports so callers can write
#   from memory_core.cache import embedding_cache_get, embedding_cache_put
# without knowing the helper module layout.

from memory_core.embedding_cache import (  # noqa: E402, F401
    cache_key as _v11_embedding_cache_key,
    get as embedding_cache_get,
    put as embedding_cache_put,
    vacuum as embedding_cache_vacuum,
    stats as embedding_cache_stats,
)


__all__ = [
    "CacheLayer",
    "TwoLevelCache",
    "L1QueryCache",
    "L2EmbeddingCache",
    "LRU",
    "embedding_cache_key",
    "make_l1_key",
    "make_l2_key",
    # v11 persistent cache
    "embedding_cache_get",
    "embedding_cache_put",
    "embedding_cache_vacuum",
    "embedding_cache_stats",
]
