"""v11 W1-A — Episode layer data types.

Plain dataclasses, no Pydantic dependency. Episodes are emitted by the
extractor and returned by the retriever; both surfaces share the same
field names so downstream consumers (recall fusion, eval harness) can
treat them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EpisodeFact:
    """A single fact (knowledge row) referenced by an episode."""

    knowledge_id: int
    created_at: str            # ISO 8601 of the source fact
    content: str               # raw fact text (used for summary fallback)
    tags: tuple[str, ...] = ()  # canonical entity / topic tags


@dataclass
class EpisodeRecord:
    """One coherent (when, who, where, what, why, outcome) unit.

    `id` is None until the row has been written to SQLite. After insert
    the extractor sets it to the autoincrement value SQLite returns so
    callers can correlate episode_facts links.
    """

    project: str
    started_at: str            # ISO 8601, first fact in the segment
    ended_at: str              # ISO 8601, last fact in the segment
    summary: str               # compact narrative (LLM or deterministic fallback)
    session_id: str | None = None
    participants: tuple[str, ...] = ()
    location: str | None = None
    outcome: str | None = None
    fact_ids: tuple[int, ...] = ()
    embedding: tuple[float, ...] | None = None
    id: int | None = None

    def to_row(self) -> dict[str, object]:
        """Mapping ready for sqlite3 named-parameter INSERT."""
        return {
            "project": self.project,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "participants": _json_tuple(self.participants),
            "location": self.location,
            "summary": self.summary,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class EpisodeHit:
    """One retrieval result.

    `score` is the fused RRF score; `bm25_rank` and `cosine_rank` are the
    underlying per-channel ranks (1-based, missing channel = None) so
    callers can inspect why a row surfaced.
    """

    episode_id: int
    score: float
    summary: str
    started_at: str
    ended_at: str
    fact_ids: tuple[int, ...] = ()
    project: str = ""
    session_id: str | None = None
    bm25_rank: int | None = None
    cosine_rank: int | None = None


# ─── helpers ───────────────────────────────────────────────────────────


def _json_tuple(values: tuple[str, ...] | list[str] | None) -> str | None:
    if not values:
        return None
    import json
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


__all__ = ["EpisodeRecord", "EpisodeHit", "EpisodeFact"]
