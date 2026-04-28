"""v11 W1-A — Episode layer.

A first-class Episode is a coherent (when, who, where, what, why,
outcome) unit composed from a contiguous run of facts in the flat
`knowledge` store. The layer is intentionally narrow:

* :mod:`schema`   — dataclasses for Episode rows and retriever hits.
* :mod:`extractor`— builds episodes from a session's facts.
* :mod:`retriever`— BM25 + cosine RRF lookup over episodes.

All three modules are pure-data: they do NOT import server.py and they
do NOT mutate `knowledge` rows. This keeps the legacy hot path
(server.Store, recall.py) unaffected until a later phase wires the
retriever into the recall ladder.
"""

from .schema import EpisodeRecord, EpisodeHit, EpisodeFact
from .extractor import extract_episodes_from_session
from .retriever import retrieve_episodes

__all__ = [
    "EpisodeRecord",
    "EpisodeHit",
    "EpisodeFact",
    "extract_episodes_from_session",
    "retrieve_episodes",
]
