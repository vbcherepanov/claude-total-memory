"""Tests for the v11 W1-A episode retriever.

The retriever fuses a BM25 channel (FTS5 over the episode summary) with
a cosine channel (over the stored summary embedding) using Reciprocal
Rank Fusion, k = 60. These tests insert a curated batch of fake
episodes and verify:

* The fused top-1 matches the query intent.
* RRF fusion is at least as good as BM25-only on a query designed to
  punish lexical-only retrieval (synonyms with no token overlap).
"""

from __future__ import annotations

import math
import sqlite3
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from memory_core.episodes import retrieve_episodes  # noqa: E402


_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_BASE_DDL = """
CREATE TABLE IF NOT EXISTS migrations (
    version TEXT PRIMARY KEY,
    description TEXT,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


# ─── deterministic concept embedder ─────────────────────────────────────


_CONCEPTS = (
    "auth", "billing", "deploy", "search", "logging",
    "metrics", "cache", "queue", "graph", "scheduler",
    "secrets", "rotation", "tokens",
)
_CONCEPT_INDEX = {w: i for i, w in enumerate(_CONCEPTS)}
_DIM = len(_CONCEPTS)

# Synonym mapping: words that are not in _CONCEPTS but should still
# steer the embedding toward an existing axis. This is what gives the
# semantic channel an edge over BM25 on lexical-mismatch queries.
_SYNONYMS = {
    "credential": "auth",
    "credentials": "auth",
    "login": "auth",
    "jwt": "auth",
    "session": "auth",
    "passkey": "auth",
    "stripe": "billing",
    "invoice": "billing",
    "subscription": "billing",
    "kubernetes": "deploy",
    "docker": "deploy",
    "rollout": "deploy",
    "elasticsearch": "search",
    "index": "search",
    "lucene": "search",
}


def _embed(text: str) -> list[float]:
    vec = [0.0] * _DIM
    if not text:
        return vec
    lower = text.lower()
    for word, idx in _CONCEPT_INDEX.items():
        if word in lower:
            vec[idx] += 1.0
    for word, target in _SYNONYMS.items():
        if word in lower:
            vec[_CONCEPT_INDEX[target]] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ─── fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_BASE_DDL)
    migration = (_MIGRATIONS_DIR / "023_episodes.sql").read_text()
    conn.executescript(migration)
    yield conn
    conn.close()


def _ts(base: datetime, minutes: float) -> str:
    return (base + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _insert_episode(
    conn: sqlite3.Connection,
    *,
    project: str,
    summary: str,
    started_at: str,
    ended_at: str,
    participants: tuple[str, ...] = (),
    session_id: str | None = None,
    fact_ids: tuple[int, ...] = (),
) -> int:
    blob = _vec_to_blob(_embed(summary))
    cur = conn.execute(
        """
        INSERT INTO episodes_v11
            (project, session_id, started_at, ended_at,
             participants, location, summary, outcome, embedding_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project, session_id, started_at, ended_at,
            ",".join(participants) if participants else None,
            None, summary, None, blob,
        ),
    )
    eid = int(cur.lastrowid)
    conn.execute(
        "INSERT INTO episodes_v11_fts (rowid, summary, participants, outcome) "
        "VALUES (?, ?, ?, ?)",
        (eid, summary, " ".join(participants), ""),
    )
    for kid in fact_ids:
        conn.execute(
            "INSERT INTO episode_facts (episode_id, knowledge_id) VALUES (?, ?)",
            (eid, int(kid)),
        )
    conn.commit()
    return eid


@pytest.fixture
def populated_db(db: sqlite3.Connection) -> sqlite3.Connection:
    base = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)
    project = "alpha"
    summaries = [
        ("Rotated JWT signing keys and updated auth middleware", ("auth",)),
        ("Investigated stripe invoice webhook double-fire", ("billing",)),
        ("Configured docker kubernetes rollout for staging", ("deploy",)),
        ("Tuned elasticsearch index mappings for product search", ("search",)),
        ("Wired structured logging across services", ("logging",)),
        ("Exposed prometheus metrics endpoints", ("metrics",)),
        ("Added redis cache layer in front of postgres", ("cache",)),
        ("Set up rabbitmq queue with dead-letter routing", ("queue",)),
        ("Modeled order graph with adjacency tables", ("graph",)),
        ("Built nightly scheduler for report aggregation", ("scheduler",)),
    ]
    for i, (text, parts) in enumerate(summaries):
        _insert_episode(
            db, project=project, summary=text,
            started_at=_ts(base, i * 60),
            ended_at=_ts(base, i * 60 + 30),
            participants=parts,
            session_id=f"sess-{i}",
            fact_ids=(i * 10 + 1, i * 10 + 2),
        )
    return db


# ─── tests ──────────────────────────────────────────────────────────────


def test_returns_empty_for_blank_query(populated_db):
    out = retrieve_episodes(populated_db, "", "alpha", embed_fn=_embed)
    assert out == []
    out = retrieve_episodes(populated_db, "   ", "alpha", embed_fn=_embed)
    assert out == []


def test_top1_lexical_match(populated_db):
    hits = retrieve_episodes(
        populated_db, "stripe invoice webhook", "alpha", k=3, embed_fn=_embed,
    )
    assert len(hits) >= 1
    assert "stripe" in hits[0].summary.lower()
    assert hits[0].project == "alpha"
    assert hits[0].fact_ids  # link table hydrated


def test_project_filter(populated_db):
    # Insert one episode under a different project — must not surface
    base = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    _insert_episode(
        populated_db, project="beta",
        summary="Stripe invoice double-fire fix",
        started_at=_ts(base, 0), ended_at=_ts(base, 5),
        participants=("billing",),
    )
    hits = retrieve_episodes(
        populated_db, "stripe invoice webhook", "alpha", k=5, embed_fn=_embed,
    )
    assert all(h.project == "alpha" for h in hits)


def test_fact_ids_hydrated(populated_db):
    hits = retrieve_episodes(
        populated_db, "kubernetes rollout staging", "alpha", k=1, embed_fn=_embed,
    )
    assert len(hits) == 1
    assert hits[0].fact_ids == (21, 22)


def test_rrf_recovers_synonym_match(populated_db):
    """Query uses 'credential rotation' — no token overlap with any
    summary, but the cosine channel maps it onto the 'auth' axis via
    the synonym table. RRF fusion must surface the JWT rotation episode
    at top-1; pure BM25 cannot.
    """
    query = "credential rotation playbook"

    # Pure BM25: simulate by calling the retriever with an embedder that
    # returns an all-zero vector ⇒ cosine channel contributes nothing.
    def _zero_embed(_text: str) -> list[float]:
        return [0.0] * _DIM

    bm25_hits = retrieve_episodes(
        populated_db, query, "alpha", k=3, embed_fn=_zero_embed,
    )
    fused_hits = retrieve_episodes(
        populated_db, query, "alpha", k=3, embed_fn=_embed,
    )

    assert fused_hits, "RRF retrieval should return something"
    # Cosine alone surfaces the JWT/auth episode for this query
    assert "jwt" in fused_hits[0].summary.lower()

    # BM25-only either misses entirely or surfaces something unrelated
    bm25_top_summary = bm25_hits[0].summary.lower() if bm25_hits else ""
    assert "jwt" not in bm25_top_summary

    # And the fused hit carries a cosine_rank ⇒ proof the cosine channel
    # contributed to the win.
    assert fused_hits[0].cosine_rank is not None


def test_global_search_when_project_is_none(populated_db):
    base = datetime(2026, 4, 3, 9, 0, tzinfo=timezone.utc)
    _insert_episode(
        populated_db, project="gamma",
        summary="Stripe invoice anomaly investigation",
        started_at=_ts(base, 0), ended_at=_ts(base, 10),
        participants=("billing",),
    )
    hits = retrieve_episodes(
        populated_db, "stripe invoice", project=None, k=5, embed_fn=_embed,
    )
    projects = {h.project for h in hits}
    assert "alpha" in projects
    assert "gamma" in projects


def test_k_is_respected(populated_db):
    hits = retrieve_episodes(
        populated_db, "auth billing deploy search logging metrics cache queue graph scheduler",
        "alpha", k=3, embed_fn=_embed,
    )
    assert len(hits) <= 3


def test_score_is_descending(populated_db):
    hits = retrieve_episodes(
        populated_db, "elasticsearch product search index", "alpha",
        k=5, embed_fn=_embed,
    )
    assert len(hits) >= 2
    for a, b in zip(hits, hits[1:]):
        assert a.score >= b.score


def test_zero_k_returns_empty(populated_db):
    out = retrieve_episodes(populated_db, "stripe", "alpha", k=0, embed_fn=_embed)
    assert out == []


def test_unknown_project_returns_empty(populated_db):
    out = retrieve_episodes(
        populated_db, "stripe", "no-such-project", k=5, embed_fn=_embed,
    )
    assert out == []
