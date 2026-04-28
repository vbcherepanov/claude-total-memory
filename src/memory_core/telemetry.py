"""v11.0 Phase 3 — In-process telemetry.

Tiny counter + timer surface used by hot-path code. Deliberately not
Prometheus — that wiring lives in `src/metrics/` and consumes these
counters via :func:`snapshot`.

Counters:
  llm_calls       — must stay 0 in fast mode (the v11 invariant).
  network_calls   — same; non-localhost or :11434 traffic = a violation.
  fts_ms          — total milliseconds spent in FTS5 lookups.
  embed_ms        — total ms in embedding model calls.
  vector_ms       — total ms in vector store search/add.
  save_total_ms   — wall-clock ms accumulated across `Store.save_knowledge`.
  search_total_ms — wall-clock ms accumulated across `Recall.search`.

Usage:

    from memory_core.telemetry import counters, op_timer

    counters.bump("fts_ms_calls")
    with op_timer("embed_ms"):
        provider.embed_texts([...])
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator


class Counter:
    """Thread-safe in-process counter / timer accumulator."""

    __slots__ = ("_data", "_lock")

    def __init__(self) -> None:
        # Pre-seed canonical names so `snapshot()` always lists them.
        self._data: dict[str, float] = {
            "llm_calls": 0.0,
            "network_calls": 0.0,
            "fts_ms": 0.0,
            "embed_ms": 0.0,
            "vector_ms": 0.0,
            "save_total_ms": 0.0,
            "search_total_ms": 0.0,
        }
        self._lock = threading.Lock()

    def bump(self, name: str, delta: float = 1.0) -> None:
        with self._lock:
            self._data[name] = self._data.get(name, 0.0) + float(delta)

    def add(self, name: str, value: float) -> None:
        self.bump(name, value)

    def get(self, name: str) -> float:
        with self._lock:
            return float(self._data.get(name, 0.0))

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._data)

    def reset(self) -> None:
        with self._lock:
            for k in list(self._data.keys()):
                self._data[k] = 0.0


# Module-level singleton — every memory_core caller shares it.
counters = Counter()


@contextmanager
def op_timer(name: str) -> Iterator[None]:
    """Context manager that records elapsed milliseconds into `counters`.

    The `name` should end with `_ms` for clarity (e.g. `embed_ms`); we
    don't enforce it because some callers use bare verbs.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        counters.bump(name, elapsed_ms)


__all__ = ["Counter", "counters", "op_timer"]
