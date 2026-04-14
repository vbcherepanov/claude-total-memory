"""Integration test: memory_save → triple_extraction_queue enqueue."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    """Instantiate real Store on a fresh temp MEMORY_DIR."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    # Each test gets its own directory — avoid touching prod data.
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)

    import server  # imported lazily so MEMORY_DIR override sticks
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)

    s = server.Store()
    yield s
    try:
        s.db.close()
    except Exception:
        pass


def test_memory_save_enqueues_for_triple_extraction(store):
    # Need an active session
    import server as _srv

    sid = "sess-" + "1" * 8
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES (?, ?, 'demo', 'open')",
        (sid, "2026-04-14T00:00:00Z"),
    )
    store.db.commit()

    rid, _dup, _red = store.save_knowledge(
        sid=sid,
        content="User prefers Go for backend services built with gRPC.",
        ktype="fact",
        project="demo",
        tags=["go", "backend"],
    )
    assert rid

    # Should have a pending row in triple_extraction_queue
    row = store.db.execute(
        "SELECT status, knowledge_id FROM triple_extraction_queue WHERE knowledge_id=?",
        (rid,),
    ).fetchone()
    assert row is not None
    assert row["status"] == "pending"
    assert row["knowledge_id"] == rid


def test_memory_save_survives_queue_failure(store, monkeypatch):
    """If enqueue raises, save_knowledge should still succeed."""
    import triple_extraction_queue as teq

    original = teq.TripleExtractionQueue.enqueue

    def boom(self, knowledge_id):
        raise RuntimeError("queue down")

    monkeypatch.setattr(teq.TripleExtractionQueue, "enqueue", boom)

    sid = "sess-" + "2" * 8
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES (?, ?, 'demo', 'open')",
        (sid, "2026-04-14T00:00:00Z"),
    )
    store.db.commit()

    rid, _, _ = store.save_knowledge(
        sid=sid, content="any content for testing", ktype="fact", project="demo"
    )
    assert rid  # save succeeded despite queue crash

    # Restore for subsequent tests
    monkeypatch.setattr(teq.TripleExtractionQueue, "enqueue", original)
