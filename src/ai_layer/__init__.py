"""v11.0 AI Layer — every LLM/Ollama/CrossEncoder code path lives here.

Hot-path code in `memory_core` MUST NOT import this package. The
asymmetry is intentional:

    memory_core   → deterministic, sync, zero LLM, zero network.
    ai_layer      → may import llm_provider, ollama, anthropic,
                    cross-encoder, FlagEmbedding, etc.

The async enrichment worker (`ai_layer.enrichment_worker`) is the only
path through which an LLM/Ollama call may be triggered as a side-effect
of a `memory_save`. Direct callers of LLM helpers (representations,
deep_enricher, contradiction_detector, reflection, self_improve) live
here too so that grep/AST tools can answer "what touches the LLM?" with
a single directory walk.

The regression suite (`tests/test_v11_layer_separation.py`) enforces the
import wall: any `from ai_layer.*` reference inside `src/memory_core/`
will fail CI.

Submodules
----------

    enrichment_worker          v10.1 inbox/outbox worker (drain, enqueue, run_pending)
    enrichment_jobs            unified enqueue() shim across the three v10 queues
    summarizer                 representations.summary
    keyword_extractor          representations.keywords
    question_generator         representations.utility_questions
    relation_extractor         deep_enricher concept/triple extraction
    contradiction_detector     auto-supersession of decisions/solutions
    reflection                 ReflectionAgent + run_full
    self_improve               error/fix logging + pattern checks

The following four LLM-touching modules are also reachable through this
package so the layer-separation test sees them:

    quality_gate, coref_resolver, reranker, query_rewriter

They are imported by `enrichment_worker` (quality_gate, coref_resolver)
or by Recall (reranker, query_rewriter) but, by living under `ai_layer`,
they are guaranteed to be invisible to anything that imports
`memory_core`.
"""

from __future__ import annotations

from . import (  # noqa: F401  (re-export entry points)
    contradiction_detector,
    coref_resolver,
    enrichment_jobs,
    enrichment_worker,
    keyword_extractor,
    quality_gate,
    query_rewriter,
    question_generator,
    reflection,
    relation_extractor,
    reranker,
    self_improve,
    summarizer,
)

__all__ = [
    "contradiction_detector",
    "coref_resolver",
    "enrichment_jobs",
    "enrichment_worker",
    "keyword_extractor",
    "quality_gate",
    "query_rewriter",
    "question_generator",
    "reflection",
    "relation_extractor",
    "reranker",
    "self_improve",
    "summarizer",
]
