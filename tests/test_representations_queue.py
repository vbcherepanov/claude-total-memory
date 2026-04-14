"""Tests for representations_queue — async multi-repr generation pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def repq_db():
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
    conn.executescript((root / "migrations" / "002_multi_representation.sql").read_text())
    conn.executescript((root / "migrations" / "005_representations_queue.sql").read_text())
    yield conn
    conn.close()


def _add(db, content: str = "doc") -> int:
    return db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, ?)",
        (content, "2026-04-14T00:00:00Z"),
    ).lastrowid


def test_enqueue_idempotent(repq_db):
    from representations_queue import RepresentationsQueue

    q = RepresentationsQueue(repq_db)
    kid = _add(repq_db)
    assert q.enqueue(kid) is True
    assert q.enqueue(kid) is False
    cnt = repq_db.execute(
        "SELECT COUNT(*) FROM representations_queue WHERE knowledge_id=?", (kid,)
    ).fetchone()[0]
    assert cnt == 1


def test_process_writes_all_representations(repq_db):
    """Worker calls generator + embedder, writes one row per representation."""
    from representations_queue import RepresentationsQueue

    q = RepresentationsQueue(repq_db)
    kid = _add(repq_db, "Long content for generating four representations of.")
    q.enqueue(kid)

    def fake_generator(content: str, project: str | None = None) -> dict[str, str]:
        return {
            "summary": "short summary",
            "keywords": "alpha, beta, gamma",
            "questions": "Q1?\nQ2?",
        }

    def fake_embedder(text: str) -> list[float]:
        # Unique but deterministic embedding per text
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [b / 255.0 for b in h[:8]]

    stats = q.process_pending(
        generator=fake_generator, embedder=fake_embedder, model_name="fake", limit=5
    )
    assert stats["processed"] == 1

    # Expected: 4 rows — raw + summary + keywords + questions
    rows = repq_db.execute(
        "SELECT representation, content FROM knowledge_representations "
        "WHERE knowledge_id=? ORDER BY representation",
        (kid,),
    ).fetchall()
    kinds = {r["representation"] for r in rows}
    assert kinds == {"raw", "summary", "keywords", "questions"}


def test_process_skips_empty_generator_output(repq_db):
    """If generator returns empty dict (e.g. all LLM calls failed), raw still stored."""
    from representations_queue import RepresentationsQueue

    q = RepresentationsQueue(repq_db)
    kid = _add(repq_db, "doc")
    q.enqueue(kid)

    stats = q.process_pending(
        generator=lambda c, **_: {"summary": "", "keywords": "", "questions": ""},
        embedder=lambda t: [0.1, 0.2, 0.3],
        model_name="fake",
        limit=1,
    )
    assert stats["processed"] == 1

    kinds = {
        r["representation"]
        for r in repq_db.execute(
            "SELECT representation FROM knowledge_representations WHERE knowledge_id=?",
            (kid,),
        ).fetchall()
    }
    # At minimum raw embedding stored
    assert "raw" in kinds
    # Empty generator outputs are skipped (no summary/keywords/questions written)
    assert "summary" not in kinds


def test_process_handles_generator_crash(repq_db):
    from representations_queue import RepresentationsQueue

    q = RepresentationsQueue(repq_db, max_attempts=2)
    kid = _add(repq_db, "doc")
    q.enqueue(kid)

    def boom(content, project=None):
        raise RuntimeError("ollama died")

    stats = q.process_pending(
        generator=boom,
        embedder=lambda t: [0.0] * 4,
        model_name="fake",
        limit=1,
    )
    assert stats["failed"] == 1

    row = repq_db.execute(
        "SELECT status FROM representations_queue WHERE knowledge_id=?", (kid,)
    ).fetchone()
    # Retryable
    assert row["status"] == "pending"


def test_stats(repq_db):
    from representations_queue import RepresentationsQueue

    q = RepresentationsQueue(repq_db)
    k1 = _add(repq_db, "a")
    k2 = _add(repq_db, "b")
    q.enqueue(k1)
    q.enqueue(k2)
    q.process_pending(
        generator=lambda c, **_: {"summary": "s"},
        embedder=lambda t: [0.1, 0.2],
        model_name="fake",
        limit=1,
    )
    s = q.stats()
    assert s["pending"] == 1
    assert s["done"] == 1
