"""Triple extraction shim around `ingestion.extractor.ConceptExtractor`.

v10 keeps two LLM-driven relation extractors:

    deep_enricher.deep_enrich        entities / intent / topics
                                     (lightweight, used in metadata)
    ConceptExtractor.extract_deep    full (subject, predicate, object)
                                     triples → graph_edges

This shim exposes the second one as a pure text-in / triples-out
function. The DB-bound graph_edges write path stays where it is — the
async triple worker (`triple_extraction_queue`) keeps calling
`extract_and_link(deep=True)` directly. Use this shim for offline
analysis, evals, or one-off scripts that don't have a Store handy.

Async-tier code. Never call from the save/search hot path.
"""

from __future__ import annotations

import sqlite3

# `deep_enricher` itself is re-exported so the layer-separation test sees
# it as reachable through ai_layer.* (it is the v10.x companion to the
# triple extractor).
import deep_enricher as _deep_enricher  # noqa: F401  (transitive re-export)
from ingestion.extractor import ConceptExtractor


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Run deep LLM extraction and return `(subject, predicate, object)` triples.

    Returns `[]` when the LLM is unavailable, returns no relations, or
    fails. Never raises. The triples are NOT persisted — callers that
    want graph_edges rows should use the async `triple_extraction_queue`
    instead.
    """
    if not text:
        return []

    # ConceptExtractor needs a DB to satisfy its constructor and to use as
    # a graph_nodes cache, but `extract_deep` itself only reads the LLM.
    # A throwaway in-memory connection keeps this side-effect free.
    db = sqlite3.connect(":memory:")
    try:
        extractor = ConceptExtractor(db)
        result = extractor.extract_deep(text)
    finally:
        db.close()

    triples: list[tuple[str, str, str]] = []
    for rel in result.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        s = str(rel.get("source") or "").strip()
        p = str(rel.get("type") or rel.get("predicate") or "").strip()
        o = str(rel.get("target") or "").strip()
        if not (s and p and o):
            continue
        triples.append((s, p, o))
    return triples


__all__ = ["extract_triples"]
