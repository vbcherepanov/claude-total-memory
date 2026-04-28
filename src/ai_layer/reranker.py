"""Re-export of `src/reranker.py` (B3-B6 of the hot-path audit).

The reranker bundles HyDE expansion, query analysis, CrossEncoder
rerank, MMR diversification, and LLM-based pair scoring. Every public
helper is LLM- or heavy-ML-bound, which is exactly why it must live
under `ai_layer`. v11 fast mode disables every entry point by default;
they only fire when `MEMORY_RERANK_ENABLED=true` (or equivalent).
"""

from __future__ import annotations

from reranker import (  # noqa: F401  (re-exports)
    _bge_rerank_scores,
    _get_bge_v2_m3,
    analyze_query,
    hyde_expand,
    mmr_diversify,
    rerank_results,
)

__all__ = [
    "analyze_query",
    "hyde_expand",
    "mmr_diversify",
    "rerank_results",
    "_bge_rerank_scores",
    "_get_bge_v2_m3",
]
