"""Tests for the v11 W1-A episode extractor.

The extractor reads facts from the flat ``knowledge`` store and segments
them into episodes whenever a time gap, topic shift, or participant
change occurs. These tests build synthetic sessions with known
boundaries and assert the segmentation matches expectations.

Embeddings are injected as a deterministic stub so the tests neither
require fastembed nor pay for a real model load.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Ensure src/ is importable for memory_core.*
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from memory_core.episodes import (  # noqa: E402
    EpisodeRecord,
    extract_episodes_from_session,
)


# ─── fixtures ───────────────────────────────────────────────────────────

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_BASE_KNOWLEDGE_DDL = """
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, type TEXT, content TEXT, context TEXT DEFAULT '',
    project TEXT DEFAULT 'general', tags TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active', confidence REAL DEFAULT 1.0,
    created_at TEXT, updated_at TEXT, recall_count INTEGER DEFAULT 0,
    last_recalled TEXT, last_confirmed TEXT, superseded_by INTEGER,
    source TEXT DEFAULT 'explicit', branch TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS migrations (
    version TEXT PRIMARY KEY,
    description TEXT,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


@pytest.fixture
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_BASE_KNOWLEDGE_DDL)
    migration = (_MIGRATIONS_DIR / "023_episodes.sql").read_text()
    conn.executescript(migration)
    yield conn
    conn.close()


@pytest.fixture
def short_summarizer():
    """Deterministic stand-in for the LLM summarizer."""
    return lambda text: text[:80]


# ─── deterministic embedder ────────────────────────────────────────────


_TOPIC_VOCAB = {
    "auth": (1.0, 0.0, 0.0, 0.0),
    "jwt": (1.0, 0.0, 0.0, 0.0),
    "login": (1.0, 0.0, 0.0, 0.0),
    "billing": (0.0, 1.0, 0.0, 0.0),
    "invoice": (0.0, 1.0, 0.0, 0.0),
    "stripe": (0.0, 1.0, 0.0, 0.0),
    "deploy": (0.0, 0.0, 1.0, 0.0),
    "docker": (0.0, 0.0, 1.0, 0.0),
    "kubernetes": (0.0, 0.0, 1.0, 0.0),
    "search": (0.0, 0.0, 0.0, 1.0),
    "index": (0.0, 0.0, 0.0, 1.0),
    "elastic": (0.0, 0.0, 0.0, 1.0),
}


def _topic_embed(text: str) -> tuple[float, ...]:
    """Tiny bag-of-words embedder over a fixed topic vocabulary.

    Lowercases the text, sums one-hot vectors for any vocabulary hit,
    L2-normalises. Words outside the vocab contribute nothing — that
    keeps cosine similarity between same-topic facts ≈ 1.0 and
    different-topic facts = 0.0, which is exactly what the segmentation
    rules need to fire.
    """
    vec = [0.0] * 4
    if text:
        lower = text.lower()
        for word, axes in _TOPIC_VOCAB.items():
            if word in lower:
                for i, val in enumerate(axes):
                    vec[i] += val
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # Map empty vectors to a neutral 5th axis so cosine stays defined.
        return (0.0, 0.0, 0.0, 0.0, 1.0)
    return tuple(v / norm for v in vec) + (0.0,)


@pytest.fixture
def embed_fn():
    return _topic_embed


# ─── helpers ────────────────────────────────────────────────────────────


def _ts(base: datetime, minutes: float) -> str:
    return (base + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _insert_fact(
    conn: sqlite3.Connection,
    *,
    project: str,
    session_id: str,
    content: str,
    created_at: str,
    tags: list[str] | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO knowledge (session_id, type, content, project, tags, status, created_at)
        VALUES (?, 'fact', ?, ?, ?, 'active', ?)
        """,
        (session_id, content, project, json.dumps(tags or []), created_at),
    )
    return int(cur.lastrowid)


# ─── tests ──────────────────────────────────────────────────────────────


def test_empty_session_returns_no_episodes(db, short_summarizer, embed_fn):
    out = extract_episodes_from_session(
        db, project="p", session_id="empty",
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert out == []


def test_single_segment_for_contiguous_facts(db, short_summarizer, embed_fn):
    project = "alpha"
    session = "s1"
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    for i in range(4):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"jwt auth refresh notes #{i}",
            created_at=_ts(base, i * 5),  # 5-min intervals
            tags=["auth"],
        )

    eps = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )

    assert len(eps) == 1
    ep = eps[0]
    assert isinstance(ep, EpisodeRecord)
    assert ep.id is not None
    assert ep.project == project
    assert ep.session_id == session
    assert len(ep.fact_ids) == 4
    assert ep.participants == ("auth",)
    # Anchor row exists in DB
    row = db.execute(
        "SELECT id, started_at, ended_at FROM episodes_v11 WHERE id = ?",
        (ep.id,),
    ).fetchone()
    assert row is not None
    assert row["started_at"] == _ts(base, 0)
    assert row["ended_at"] == _ts(base, 15)
    # Fact links written
    n_links = db.execute(
        "SELECT COUNT(*) FROM episode_facts WHERE episode_id = ?", (ep.id,),
    ).fetchone()[0]
    assert n_links == 4


def test_time_gap_opens_new_episode(db, short_summarizer, embed_fn):
    project = "alpha"
    session = "s2"
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    # First cluster: 3 facts within 10 minutes
    for i in range(3):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"jwt auth note {i}",
            created_at=_ts(base, i * 5),
            tags=["auth"],
        )
    # Second cluster: 90 minutes later, same topic — boundary still fires
    for i in range(2):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"jwt refresh note {i}",
            created_at=_ts(base, 90 + i * 3),
            tags=["auth"],
        )

    eps = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert len(eps) == 2
    assert len(eps[0].fact_ids) == 3
    assert len(eps[1].fact_ids) == 2
    assert eps[0].started_at < eps[1].started_at


def test_topic_shift_opens_new_episode(db, short_summarizer, embed_fn):
    project = "alpha"
    session = "s3"
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    # Cluster 1: auth (3 facts)
    for i in range(3):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"jwt login flow detail {i}",
            created_at=_ts(base, i * 4),
            tags=["auth"],
        )
    # Cluster 2: billing — same time window (no gap) but different topic
    for i in range(3):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"stripe invoice webhook step {i}",
            created_at=_ts(base, 12 + i * 4),
            tags=["billing"],
        )

    eps = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert len(eps) == 2
    assert len(eps[0].fact_ids) == 3
    assert len(eps[1].fact_ids) == 3
    # Participants reflect the topic split
    assert eps[0].participants == ("auth",)
    assert eps[1].participants == ("billing",)


def test_participant_change_opens_new_episode(db, short_summarizer, embed_fn):
    project = "alpha"
    session = "s4"
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    # Same topic word "auth" but different tags ⇒ participant rule kicks in.
    for i in range(2):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"jwt login {i}",
            created_at=_ts(base, i * 3),
            tags=["alice"],
        )
    for i in range(2):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"jwt login {i}",
            created_at=_ts(base, 6 + i * 3),
            tags=["bob"],
        )

    eps = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert len(eps) == 2
    assert eps[0].participants == ("alice",)
    assert eps[1].participants == ("bob",)


def test_idempotent_second_run_inserts_nothing(db, short_summarizer, embed_fn):
    project = "alpha"
    session = "s5"
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    for i in range(3):
        _insert_fact(
            db, project=project, session_id=session,
            content=f"jwt note {i}",
            created_at=_ts(base, i * 2),
            tags=["auth"],
        )
    first = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert len(first) == 1
    second = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert second == []
    total = db.execute("SELECT COUNT(*) FROM episodes_v11").fetchone()[0]
    assert total == 1


def test_fallback_summarizer_used_when_llm_absent(db, embed_fn):
    project = "alpha"
    session = "s6"
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    _insert_fact(
        db, project=project, session_id=session,
        content="JWT rotation playbook overview", created_at=_ts(base, 0),
        tags=["auth"],
    )
    _insert_fact(
        db, project=project, session_id=session,
        content="JWT rotation step two", created_at=_ts(base, 5),
        tags=["auth"],
    )
    eps = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=None, embed_fn=embed_fn,
    )
    assert len(eps) == 1
    assert "JWT rotation playbook overview" in eps[0].summary
    # Two facts joined by "\n\n" → fallback marks +1 more
    assert "+1 more" in eps[0].summary


def test_three_segment_session_mixed_boundaries(db, short_summarizer, embed_fn):
    """Exercise all three rules in one session."""
    project = "alpha"
    session = "s7"
    base = datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)
    # Seg A: auth, alice — 2 facts close in time
    _insert_fact(db, project=project, session_id=session,
                 content="jwt login alice 1", created_at=_ts(base, 0),
                 tags=["auth", "alice"])
    _insert_fact(db, project=project, session_id=session,
                 content="jwt login alice 2", created_at=_ts(base, 5),
                 tags=["auth", "alice"])
    # Seg B: billing topic shift, same minute as A end → topic rule
    _insert_fact(db, project=project, session_id=session,
                 content="stripe invoice webhook step", created_at=_ts(base, 8),
                 tags=["billing"])
    _insert_fact(db, project=project, session_id=session,
                 content="stripe invoice retry", created_at=_ts(base, 11),
                 tags=["billing"])
    # Seg C: deploy topic, 2 hours later → both topic AND time-gap rules
    _insert_fact(db, project=project, session_id=session,
                 content="docker kubernetes rollout", created_at=_ts(base, 130),
                 tags=["deploy"])

    eps = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert len(eps) == 3
    assert [len(e.fact_ids) for e in eps] == [2, 2, 1]
    assert eps[0].participants == ("alice", "auth")
    assert eps[1].participants == ("billing",)
    assert eps[2].participants == ("deploy",)


def test_extractor_persists_summary_embedding(db, short_summarizer, embed_fn):
    project = "alpha"
    session = "s8"
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    _insert_fact(db, project=project, session_id=session,
                 content="jwt note one", created_at=_ts(base, 0), tags=["auth"])
    _insert_fact(db, project=project, session_id=session,
                 content="jwt note two", created_at=_ts(base, 4), tags=["auth"])

    eps = extract_episodes_from_session(
        db, project=project, session_id=session,
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert len(eps) == 1
    blob = db.execute(
        "SELECT embedding_blob FROM episodes_v11 WHERE id = ?", (eps[0].id,),
    ).fetchone()[0]
    assert blob is not None
    # 5-dim float32 ⇒ 20 bytes
    assert len(blob) == 5 * 4


def test_project_scopes_extraction(db, short_summarizer, embed_fn):
    base = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    _insert_fact(db, project="alpha", session_id="s",
                 content="jwt alpha", created_at=_ts(base, 0), tags=["auth"])
    _insert_fact(db, project="beta", session_id="s",
                 content="jwt beta", created_at=_ts(base, 1), tags=["auth"])

    eps = extract_episodes_from_session(
        db, project="alpha", session_id="s",
        llm_summarizer=short_summarizer, embed_fn=embed_fn,
    )
    assert len(eps) == 1
    assert eps[0].project == "alpha"
    # No alpha episode wrote a beta row
    rows = db.execute("SELECT project FROM episodes_v11").fetchall()
    assert [r[0] for r in rows] == ["alpha"]


def test_extractor_requires_project_and_session(db, short_summarizer, embed_fn):
    with pytest.raises(ValueError):
        extract_episodes_from_session(
            db, project="", session_id="x",
            llm_summarizer=short_summarizer, embed_fn=embed_fn,
        )
    with pytest.raises(ValueError):
        extract_episodes_from_session(
            db, project="alpha", session_id="",
            llm_summarizer=short_summarizer, embed_fn=embed_fn,
        )
