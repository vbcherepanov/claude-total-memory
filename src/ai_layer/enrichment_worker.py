"""ai_layer shim around the v10.1 async enrichment worker.

This is the canonical worker described in §1.2 of the v11 architecture:
the inbox/outbox process that runs the heavy LLM-bound stages
(quality gate, entity-dedup audit, contradiction detector, episodic
event linking, wiki auto-refresh) off the save hot path.

Nothing here re-implements the worker — it is a thin re-export of
`src.enrichment_worker`. The shim exists so callers (server.py, tests,
future schedulers) can import from `ai_layer` and so that the v11
layer-separation regression test can prove the worker lives outside
`memory_core`.

`drain` is exposed as an alias of `run_pending` for forward-compat
naming; the underlying behaviour is identical.
"""

from __future__ import annotations

from enrichment_worker import (  # noqa: F401  (re-exports)
    EnrichmentTask,
    _enabled,
    enqueue,
    process_task,
    reclaim_stale,
    run_pending,
    start_worker,
)

# Forward-compat alias. Callers that prefer the v11 vocabulary use
# `ai_layer.enrichment_worker.drain(db)`; the v10.1 implementation name
# (`run_pending`) keeps working for everyone else.
drain = run_pending

__all__ = [
    "EnrichmentTask",
    "_enabled",
    "drain",
    "enqueue",
    "process_task",
    "reclaim_stale",
    "run_pending",
    "start_worker",
]
