"""Re-export of `src/contradiction_detector.py`.

The detector is the synchronous v10 LLM round-trip that A13 of the
hot-path audit identifies as one of the two LLM calls every save of a
`decision`/`solution` performed under v10.5 defaults. v11 routes those
calls through the async enrichment worker — but the public API
(`detect_contradictions`, `apply_supersession`, `apply_and_log`,
`ContradictionVerdict`, `should_run`) stays unchanged.

This shim exists so the v11 layer-separation regression test sees the
detector under `ai_layer.*` and so future callers (worker, CLI, tests)
can import it from a stable path.
"""

from __future__ import annotations

from contradiction_detector import (  # noqa: F401  (re-exports)
    ContradictionVerdict,
    apply_and_log,
    apply_supersession,
    detect_contradictions,
    log_verdict,
    production_candidates_query,
    production_llm_call,
    should_run,
)

__all__ = [
    "ContradictionVerdict",
    "apply_and_log",
    "apply_supersession",
    "detect_contradictions",
    "log_verdict",
    "production_candidates_query",
    "production_llm_call",
    "should_run",
]
