"""Tests for enrichment-based filtering of knowledge_ids."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def filt_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    root = Path(__file__).parent.parent
    conn.executescript((root / "migrations" / "001_v5_schema.sql").read_text())
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            project TEXT DEFAULT 'general',
            status TEXT DEFAULT 'active',
            created_at TEXT
        );
        """
    )
    conn.executescript((root / "migrations" / "004_deep_enrichment.sql").read_text())
    yield conn
    conn.close()


def _add_k(db, content: str = "x") -> int:
    return db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, '2026-04-14T00:00:00Z')",
        (content,),
    ).lastrowid


def _add_enr(
    db,
    knowledge_id: int,
    entities: list | None = None,
    intent: str = "unknown",
    topics: list | None = None,
) -> None:
    db.execute(
        """INSERT OR REPLACE INTO knowledge_enrichment
             (knowledge_id, entities, intent, topics, updated_at)
           VALUES (?, ?, ?, ?, '2026-04-14T00:00:00Z')""",
        (knowledge_id, json.dumps(entities or []), intent, json.dumps(topics or [])),
    )
    db.commit()


# ──────────────────────────────────────────────
# topic filter
# ──────────────────────────────────────────────


def test_filter_by_topic_keeps_only_matching(filt_db):
    from enrichment_filter import filter_by_enrichment

    k1 = _add_k(filt_db, "auth doc")
    k2 = _add_k(filt_db, "payments doc")
    _add_enr(filt_db, k1, topics=["auth", "security"])
    _add_enr(filt_db, k2, topics=["payments", "stripe"])

    kept = filter_by_enrichment(filt_db, [k1, k2], topics=["auth"])
    assert kept == [k1]


def test_filter_by_topic_or_semantics(filt_db):
    """Multiple topics use OR: match any."""
    from enrichment_filter import filter_by_enrichment

    k1 = _add_k(filt_db, "a")
    k2 = _add_k(filt_db, "b")
    _add_enr(filt_db, k1, topics=["auth"])
    _add_enr(filt_db, k2, topics=["billing"])

    kept = filter_by_enrichment(filt_db, [k1, k2], topics=["auth", "billing"])
    assert set(kept) == {k1, k2}


def test_filter_by_entity_matches_name(filt_db):
    from enrichment_filter import filter_by_enrichment

    k1 = _add_k(filt_db, "go stuff")
    k2 = _add_k(filt_db, "php stuff")
    _add_enr(filt_db, k1, entities=[{"name": "Go", "type": "technology"}])
    _add_enr(filt_db, k2, entities=[{"name": "PHP", "type": "technology"}])

    kept = filter_by_enrichment(filt_db, [k1, k2], entities=["Go"])
    assert kept == [k1]


def test_filter_entity_case_insensitive(filt_db):
    from enrichment_filter import filter_by_enrichment

    k = _add_k(filt_db, "doc")
    _add_enr(filt_db, k, entities=[{"name": "PostgreSQL", "type": "technology"}])

    kept = filter_by_enrichment(filt_db, [k], entities=["postgresql"])
    assert kept == [k]


def test_filter_by_intent(filt_db):
    from enrichment_filter import filter_by_enrichment

    k1 = _add_k(filt_db, "a")
    k2 = _add_k(filt_db, "b")
    _add_enr(filt_db, k1, intent="question")
    _add_enr(filt_db, k2, intent="procedural")

    kept = filter_by_enrichment(filt_db, [k1, k2], intent="question")
    assert kept == [k1]


def test_filter_combines_all_criteria_as_and(filt_db):
    """topics AND entities AND intent must all match."""
    from enrichment_filter import filter_by_enrichment

    k1 = _add_k(filt_db, "a")
    k2 = _add_k(filt_db, "b")
    _add_enr(filt_db, k1, topics=["auth"], entities=[{"name": "Go"}], intent="procedural")
    _add_enr(filt_db, k2, topics=["auth"], entities=[{"name": "Rust"}], intent="procedural")

    kept = filter_by_enrichment(
        filt_db, [k1, k2], topics=["auth"], entities=["Go"], intent="procedural"
    )
    assert kept == [k1]


def test_filter_drops_records_without_enrichment_when_filter_active(filt_db):
    """If filters are specified but a record has no enrichment row, it's excluded."""
    from enrichment_filter import filter_by_enrichment

    k1 = _add_k(filt_db, "enriched")
    k2 = _add_k(filt_db, "not enriched yet")
    _add_enr(filt_db, k1, topics=["auth"])

    kept = filter_by_enrichment(filt_db, [k1, k2], topics=["auth"])
    assert kept == [k1]


def test_no_filters_returns_all(filt_db):
    """When no criteria given, pass-through."""
    from enrichment_filter import filter_by_enrichment

    k1 = _add_k(filt_db, "a")
    k2 = _add_k(filt_db, "b")
    kept = filter_by_enrichment(filt_db, [k1, k2])
    assert set(kept) == {k1, k2}


def test_empty_candidate_list(filt_db):
    from enrichment_filter import filter_by_enrichment

    assert filter_by_enrichment(filt_db, [], topics=["auth"]) == []


def test_preserves_input_order(filt_db):
    from enrichment_filter import filter_by_enrichment

    ids = [_add_k(filt_db, f"k{i}") for i in range(5)]
    for kid in ids:
        _add_enr(filt_db, kid, topics=["common"])

    # Shuffle input — output should match input order (not DB order)
    shuffled = [ids[3], ids[0], ids[4], ids[1], ids[2]]
    kept = filter_by_enrichment(filt_db, shuffled, topics=["common"])
    assert kept == shuffled
