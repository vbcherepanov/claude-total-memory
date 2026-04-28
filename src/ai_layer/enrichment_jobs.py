"""Unified enqueue shim across the v10 enrichment queues.

v10 grew three sibling queues, each owning a slice of async LLM work:

    enrichment_queue          quality gate / entity-dedup / contradiction /
                              episodic / wiki refresh
                              → enrichment_worker.enqueue
    triple_extraction_queue   subject-predicate-object triples (graph_edges)
                              → TripleExtractionQueue.enqueue
    deep_enrichment_queue     entities / intent / topics
                              → DeepEnrichmentQueue.enqueue
    representations_queue     summary / keywords / questions / compressed
                              → RepresentationsQueue.enqueue

v11 keeps the underlying tables (no schema churn) but introduces a single
front door so `Store.save_knowledge` and the future job-scheduler do not
need to know which queue owns which job-type.

Usage:

    from ai_layer.enrichment_jobs import JobType, enqueue
    enqueue(JobType.SUMMARY,    db=db, knowledge_id=rid)
    enqueue(JobType.TRIPLES,    db=db, knowledge_id=rid)
    enqueue(JobType.REFLECTION, db=db, knowledge_id=rid)

Anything that needs richer per-job metadata (project, ktype, tags) goes
to the underlying queues directly via `ai_layer.enrichment_worker.enqueue`
— that signature is preserved.
"""

from __future__ import annotations

import enum
from typing import Any

# v10 queue classes — re-exported through ai_layer so callers don't reach
# back into the flat src/ namespace.
from deep_enrichment_queue import DeepEnrichmentQueue  # noqa: F401
from representations_queue import RepresentationsQueue  # noqa: F401
from triple_extraction_queue import TripleExtractionQueue  # noqa: F401


class JobType(str, enum.Enum):
    """All async LLM job-types known to v11."""

    SUMMARY = "summary"
    KEYWORDS = "keywords"
    QUESTIONS = "questions"
    FACTS = "facts"
    TRIPLES = "triples"
    ADVANCED_RELATIONS = "advanced_relations"
    CONTRADICTION_CHECK = "contradiction_check"
    LESSON_EXTRACTION = "lesson_extraction"
    PREFERENCE_EXTRACTION = "preference_extraction"
    PROCEDURAL_MEMORY = "procedural_memory"
    MULTI_REPRESENTATION = "multi_representation"
    REFLECTION = "reflection"


# Mapping: which underlying queue owns each job-type. Several v11 job-types
# fan out to the same v10 queue (e.g. SUMMARY/KEYWORDS/QUESTIONS all live in
# representations_queue) — that is intentional.
_QUEUE_FOR_JOB: dict[JobType, str] = {
    JobType.SUMMARY: "representations",
    JobType.KEYWORDS: "representations",
    JobType.QUESTIONS: "representations",
    JobType.MULTI_REPRESENTATION: "representations",
    JobType.TRIPLES: "triples",
    JobType.ADVANCED_RELATIONS: "triples",
    JobType.FACTS: "deep_enrichment",
    JobType.LESSON_EXTRACTION: "deep_enrichment",
    JobType.PREFERENCE_EXTRACTION: "deep_enrichment",
    JobType.PROCEDURAL_MEMORY: "deep_enrichment",
    JobType.CONTRADICTION_CHECK: "enrichment",
    JobType.REFLECTION: "enrichment",
}


def enqueue(
    job_type: JobType | str,
    *,
    db: Any,
    knowledge_id: int,
    **payload: Any,
) -> bool:
    """Route `(job_type, knowledge_id)` to the v10 queue that owns it.

    Returns True if a new pending row was inserted, False if the queue
    de-duplicated against an existing pending row.

    Extra `payload` keys are forwarded to `enrichment_worker.enqueue`
    (which has a richer signature) and ignored by the three lighter
    queues. This keeps the call-site uniform.
    """
    if isinstance(job_type, str):
        job_type = JobType(job_type)

    family = _QUEUE_FOR_JOB[job_type]

    if family == "representations":
        return RepresentationsQueue(db).enqueue(knowledge_id)
    if family == "triples":
        return TripleExtractionQueue(db).enqueue(knowledge_id)
    if family == "deep_enrichment":
        return DeepEnrichmentQueue(db).enqueue(knowledge_id)
    # `family == "enrichment"` — the v10.1 inbox/outbox worker.
    from ai_layer.enrichment_worker import enqueue as _enrichment_enqueue

    row_id = _enrichment_enqueue(
        db,
        knowledge_id=knowledge_id,
        session_id=payload.get("session_id"),
        project=payload.get("project", ""),
        ktype=payload.get("ktype", ""),
        content_snapshot=payload.get("content_snapshot", ""),
        tags_snapshot=payload.get("tags_snapshot"),
        importance=payload.get("importance", "medium"),
        skip_quality=payload.get("skip_quality", False),
    )
    return bool(row_id)


__all__ = [
    "DeepEnrichmentQueue",
    "JobType",
    "RepresentationsQueue",
    "TripleExtractionQueue",
    "enqueue",
]
