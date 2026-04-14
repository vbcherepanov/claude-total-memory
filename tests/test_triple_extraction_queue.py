"""Tests for triple extraction queue — async pipeline from memory_save to KG."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def queue_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    root = Path(__file__).parent.parent
    conn.executescript((root / "migrations" / "001_v5_schema.sql").read_text())
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT
        );
        """
    )
    conn.executescript((root / "migrations" / "003_triple_extraction_queue.sql").read_text())
    yield conn
    conn.close()


def _add_knowledge(db, content: str = "sample") -> int:
    return db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, ?)",
        (content, "2026-04-14T00:00:00Z"),
    ).lastrowid


# ──────────────────────────────────────────────
# enqueue
# ──────────────────────────────────────────────


def test_enqueue_creates_pending_row(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)

    rows = queue_db.execute(
        "SELECT status, knowledge_id FROM triple_extraction_queue"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "pending"
    assert rows[0]["knowledge_id"] == kid


def test_enqueue_idempotent(queue_db):
    """Re-enqueueing while pending must not create duplicates."""
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)
    q.enqueue(kid)
    q.enqueue(kid)

    cnt = queue_db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE knowledge_id=? AND status='pending'",
        (kid,),
    ).fetchone()[0]
    assert cnt == 1


def test_enqueue_allowed_after_done(queue_db):
    """Once done, a later enqueue (e.g. on content update) creates a new pending row."""
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)
    claimed = q.claim_next()
    q.mark_done(claimed["id"])
    q.enqueue(kid)  # new pending

    cnt = queue_db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE knowledge_id=?",
        (kid,),
    ).fetchone()[0]
    assert cnt == 2


# ──────────────────────────────────────────────
# claim_next
# ──────────────────────────────────────────────


def test_claim_next_returns_fifo(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    k1 = _add_knowledge(queue_db, "first")
    k2 = _add_knowledge(queue_db, "second")
    q.enqueue(k1)
    q.enqueue(k2)

    first = q.claim_next()
    assert first["knowledge_id"] == k1
    assert first["status"] == "processing"

    second = q.claim_next()
    assert second["knowledge_id"] == k2


def test_claim_next_none_when_empty(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    assert q.claim_next() is None


def test_claim_marks_processing_not_visible_to_other_claim(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)
    a = q.claim_next()
    b = q.claim_next()  # nothing else pending
    assert a is not None
    assert b is None


# ──────────────────────────────────────────────
# mark_done / mark_failed
# ──────────────────────────────────────────────


def test_mark_done_transitions_status(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)
    item = q.claim_next()
    q.mark_done(item["id"])

    row = queue_db.execute(
        "SELECT status, processed_at FROM triple_extraction_queue WHERE id=?",
        (item["id"],),
    ).fetchone()
    assert row["status"] == "done"
    assert row["processed_at"] is not None


def test_mark_failed_records_error_and_increments_attempts(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)
    item = q.claim_next()
    q.mark_failed(item["id"], "ollama timeout")

    row = queue_db.execute(
        "SELECT status, attempts, last_error FROM triple_extraction_queue WHERE id=?",
        (item["id"],),
    ).fetchone()
    # attempts bumped; item returns to pending for retry unless over max
    assert row["attempts"] == 1
    assert "ollama" in row["last_error"].lower()


def test_mark_failed_gives_up_after_max_attempts(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db, max_attempts=2)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)
    item = q.claim_next()
    q.mark_failed(item["id"], "err1")
    item = q.claim_next()  # re-claimed
    q.mark_failed(item["id"], "err2")

    row = queue_db.execute(
        "SELECT status, attempts FROM triple_extraction_queue WHERE id=?",
        (item["id"],),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 2


# ──────────────────────────────────────────────
# process_pending (worker)
# ──────────────────────────────────────────────


def test_process_pending_calls_extractor_and_marks_done(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db, "text to extract from")
    q.enqueue(kid)

    calls: list[int] = []

    def fake_extract(knowledge_id: int, content: str) -> dict:
        calls.append(knowledge_id)
        return {"relations": [{"source": "a", "target": "b", "type": "uses"}]}

    stats = q.process_pending(fake_extract, limit=5)
    assert stats["processed"] == 1
    assert stats["failed"] == 0
    assert calls == [kid]


def test_process_pending_respects_limit(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    for i in range(5):
        q.enqueue(_add_knowledge(queue_db, f"row{i}"))

    calls: list[int] = []

    def fake_extract(knowledge_id: int, content: str) -> dict:
        calls.append(knowledge_id)
        return {}

    stats = q.process_pending(fake_extract, limit=3)
    assert stats["processed"] == 3
    assert len(calls) == 3


def test_process_pending_handles_extractor_exception(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db, max_attempts=3)
    kid = _add_knowledge(queue_db)
    q.enqueue(kid)

    def broken_extract(knowledge_id: int, content: str) -> dict:
        raise RuntimeError("ollama down")

    stats = q.process_pending(broken_extract, limit=1)
    assert stats["processed"] == 0
    assert stats["failed"] == 1
    # still retryable
    row = queue_db.execute(
        "SELECT status FROM triple_extraction_queue WHERE knowledge_id=?", (kid,)
    ).fetchone()
    assert row["status"] == "pending"  # reset for retry


def test_stats_reports_counts(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    k1 = _add_knowledge(queue_db, "a")
    k2 = _add_knowledge(queue_db, "b")
    q.enqueue(k1)
    q.enqueue(k2)
    # process one successfully
    def ok(*_a, **_kw):
        return {}

    q.process_pending(ok, limit=1)
    s = q.stats()
    assert s["pending"] == 1
    assert s["done"] == 1


# ──────────────────────────────────────────────
# Regression: re-enqueue after 'done' must not collide
# ──────────────────────────────────────────────


def test_reenqueue_after_done_does_not_violate_unique(queue_db):
    """After mark_done, a new enqueue + mark_done must succeed without UNIQUE error."""
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db)
    kid = _add_knowledge(queue_db)

    # First cycle: enqueue → claim → done
    q.enqueue(kid)
    item1 = q.claim_next()
    q.mark_done(item1["id"])

    # Second cycle (e.g. backfill_orphan_edges re-enqueues the same kid)
    q.enqueue(kid)
    item2 = q.claim_next()
    assert item2 is not None
    q.mark_done(item2["id"])  # must NOT raise UNIQUE conflict

    # Only the latest 'done' row remains (stale one cleaned up)
    dones = queue_db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE knowledge_id=? AND status='done'",
        (kid,),
    ).fetchone()[0]
    assert dones == 1


def test_reenqueue_after_failed_does_not_violate_unique(queue_db):
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(queue_db, max_attempts=1)  # first failure = failed
    kid = _add_knowledge(queue_db)

    # Cycle 1: enqueue → claim → fail
    q.enqueue(kid)
    item1 = q.claim_next()
    q.mark_failed(item1["id"], "err1")
    assert queue_db.execute(
        "SELECT status FROM triple_extraction_queue WHERE id=?", (item1["id"],)
    ).fetchone()["status"] == "failed"

    # Cycle 2: re-enqueue after fail, then claim → fail again
    q.enqueue(kid)
    item2 = q.claim_next()
    q.mark_failed(item2["id"], "err2")  # must not raise UNIQUE

    failed_count = queue_db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE knowledge_id=? AND status='failed'",
        (kid,),
    ).fetchone()[0]
    assert failed_count == 1  # only latest retained
