"""Tests for async deep enrichment queue (entities/intent/topics pipeline)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def denr_db():
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
    conn.executescript((root / "migrations" / "004_deep_enrichment.sql").read_text())
    yield conn
    conn.close()


def _add(db, content: str) -> int:
    return db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, ?)",
        (content, "2026-04-14T00:00:00Z"),
    ).lastrowid


# ──────────────────────────────────────────────
# enqueue
# ──────────────────────────────────────────────


def test_enqueue_creates_pending(denr_db):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(denr_db)
    kid = _add(denr_db, "doc")
    assert q.enqueue(kid) is True

    row = denr_db.execute(
        "SELECT status FROM deep_enrichment_queue WHERE knowledge_id=?", (kid,)
    ).fetchone()
    assert row["status"] == "pending"


def test_enqueue_idempotent(denr_db):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(denr_db)
    kid = _add(denr_db, "x")
    q.enqueue(kid)
    q.enqueue(kid)
    cnt = denr_db.execute(
        "SELECT COUNT(*) FROM deep_enrichment_queue WHERE knowledge_id=?", (kid,)
    ).fetchone()[0]
    assert cnt == 1


# ──────────────────────────────────────────────
# Worker — writes enrichment
# ──────────────────────────────────────────────


def test_process_pending_writes_enrichment(denr_db):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(denr_db)
    kid = _add(denr_db, "Long document about Go microservices and auth.")
    q.enqueue(kid)

    def fake_enrich(content: str, base_metadata=None):
        return {
            "entities": [{"name": "Go", "type": "technology"}],
            "intent": "procedural",
            "topics": ["backend", "microservices"],
        }

    stats = q.process_pending(fake_enrich, limit=5)
    assert stats["processed"] == 1

    row = denr_db.execute(
        "SELECT entities, intent, topics FROM knowledge_enrichment WHERE knowledge_id=?",
        (kid,),
    ).fetchone()
    assert row is not None
    assert row["intent"] == "procedural"
    assert "Go" in row["entities"]
    assert "backend" in row["topics"]


def test_process_pending_upserts_existing_enrichment(denr_db):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(denr_db)
    kid = _add(denr_db, "doc")
    q.enqueue(kid)

    q.process_pending(
        lambda c, base_metadata=None: {"entities": [], "intent": "fact", "topics": ["t1"]},
        limit=1,
    )

    # Re-enqueue after content change
    q.enqueue(kid)
    q.process_pending(
        lambda c, base_metadata=None: {
            "entities": [{"name": "NewThing", "type": "concept"}],
            "intent": "updated",
            "topics": ["t2"],
        },
        limit=1,
    )

    row = denr_db.execute(
        "SELECT intent, topics FROM knowledge_enrichment WHERE knowledge_id=?", (kid,)
    ).fetchone()
    assert row["intent"] == "updated"
    assert "t2" in row["topics"]


def test_process_pending_handles_enricher_crash(denr_db):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(denr_db, max_attempts=2)
    kid = _add(denr_db, "doc")
    q.enqueue(kid)

    def broken(content, base_metadata=None):
        raise RuntimeError("ollama fail")

    stats = q.process_pending(broken, limit=1)
    assert stats["failed"] == 1

    # Item goes back to pending (retryable)
    row = denr_db.execute(
        "SELECT status FROM deep_enrichment_queue WHERE knowledge_id=?", (kid,)
    ).fetchone()
    assert row["status"] == "pending"


def test_process_pending_skips_missing_knowledge(denr_db):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(denr_db)
    kid = _add(denr_db, "temp")
    q.enqueue(kid)
    denr_db.execute("DELETE FROM knowledge WHERE id=?", (kid,))
    denr_db.commit()

    def enr(content, base_metadata=None):
        return {"entities": [], "intent": "unknown", "topics": []}

    stats = q.process_pending(enr, limit=5)
    assert stats["skipped"] == 1
    assert stats["processed"] == 0


def test_stats(denr_db):
    from deep_enrichment_queue import DeepEnrichmentQueue

    q = DeepEnrichmentQueue(denr_db)
    k1 = _add(denr_db, "a")
    k2 = _add(denr_db, "b")
    q.enqueue(k1)
    q.enqueue(k2)
    q.process_pending(
        lambda c, base_metadata=None: {"entities": [], "intent": "fact", "topics": []},
        limit=1,
    )
    s = q.stats()
    assert s["pending"] == 1
    assert s["done"] == 1
