"""Re-export of `src/quality_gate.py` (A6 of the hot-path audit).

The pre-save quality scorer is one of the two synchronous LLM calls v11
removes from the default save path. The implementation stays in the
v10 module; this shim only exposes it under `ai_layer.*` so the
enrichment worker (and the layer-separation test) see it here.
"""

from __future__ import annotations

from quality_gate import (  # noqa: F401  (re-exports)
    QualityScore,
    log_decision,
    score_quality,
    should_score,
)

__all__ = [
    "QualityScore",
    "log_decision",
    "score_quality",
    "should_score",
]
