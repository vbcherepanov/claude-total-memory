"""Verify memory_stats surfaces v6.0 queue + storage counters."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    yield s
    try:
        s.db.close()
    except Exception:
        pass


def test_stats_includes_v6_queue_block(store):
    import server as _srv

    recall = _srv.Recall(store)
    stats = recall.stats()

    assert "v6_queues" in stats
    assert set(stats["v6_queues"].keys()) == {
        "triple_extraction", "deep_enrichment", "representations"
    }
    for q in stats["v6_queues"].values():
        # Should always have pending/processing/done/failed keys
        assert "pending" in q
        assert "done" in q

    assert "v6_storage" in stats
    assert "representations_rows" in stats["v6_storage"]
    assert "enrichment_rows" in stats["v6_storage"]


def test_stats_counts_grow_after_save(store):
    import server as _srv

    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()
    store.save_knowledge(sid="s1", content="seed", ktype="fact", project="demo")
    store.save_knowledge(sid="s1", content="another", ktype="fact", project="demo")

    recall = _srv.Recall(store)
    stats = recall.stats()

    # Three queues each should have 2 pending items
    for name, counts in stats["v6_queues"].items():
        assert counts["pending"] == 2, f"queue {name} expected 2 pending, got {counts}"
