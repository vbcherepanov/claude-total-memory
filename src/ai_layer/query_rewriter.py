"""Re-export of `src/query_rewriter.py` (B1 of the hot-path audit).

Anthropic Haiku-driven query rewriter. Disabled by default in v11
(`MEMORY_QUERY_REWRITE=0`); only relevant when an operator opts back
in. Lives under `ai_layer` because it makes external HTTP calls.
"""

from __future__ import annotations

from query_rewriter import (  # noqa: F401  (re-exports)
    expand_for_retrieval,
    has_decomposable_intent,
    is_enabled,
    rewrite,
)

__all__ = [
    "expand_for_retrieval",
    "has_decomposable_intent",
    "is_enabled",
    "rewrite",
]
