"""Integration test: full pipeline must succeed when Ollama is unavailable.

This is the "user without Ollama installed" scenario. Saves should still work,
queues fill up, reflection drains them with Phase 5/6 producing only what's
LLM-free (raw embeddings). Triple extraction & enrichment skip silently.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    # Force LLM disabled before any module imports config
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")

    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server, config
    config._cache_clear()
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    yield s
    try:
        s.db.close()
    except Exception:
        pass


def test_save_works_without_llm(store):
    """memory_save must succeed even with no Ollama / no model."""
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()
    rid, dup, _ = store.save_knowledge(
        sid="s1", content="hello world", ktype="fact", project="demo"
    )
    assert rid
    # Queues still get the entries — they'll be drained later when LLM returns
    for tbl in ("triple_extraction_queue", "deep_enrichment_queue", "representations_queue"):
        n = store.db.execute(
            f"SELECT COUNT(*) FROM {tbl} WHERE knowledge_id=?", (rid,)
        ).fetchone()[0]
        assert n == 1


def test_recall_works_without_llm(store):
    """memory_recall returns results without HyDE expansion."""
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()
    store.save_knowledge(sid="s1", content="kubernetes operator pattern with crds", ktype="fact", project="demo")

    import server as _srv
    recall = _srv.Recall(store)
    result = recall.search(query="kubernetes operator", project="demo", limit=5)
    items = [i for g in result["results"].values() for i in g]
    # Found via FTS5/semantic, even without HyDE
    assert any("kubernetes" in (i.get("content", "") or "").lower() for i in items)


def test_reflection_drain_skips_llm_phases(store):
    """Reflection drain must complete without errors even if LLM is off."""
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()
    rid, _, _ = store.save_knowledge(
        sid="s1", content="some content here for processing", ktype="fact", project="demo"
    )

    from reflection.agent import ReflectionAgent
    agent = ReflectionAgent(store.db, embedder=lambda t: [0.1] * 8)
    report = asyncio.run(agent.run_drain())

    # Triples + enrichment skipped (deferred), representations stored only raw
    assert report["triple_extraction"].get("deferred") == "no_llm" or \
           report["triple_extraction"].get("processed") == 0
    assert report["deep_enrichment"].get("deferred") == "no_llm" or \
           report["deep_enrichment"].get("processed") == 0
    # Representations runs (no LLM = empty views, but raw still saved)
    assert "processed" in report["representations"]


def test_stats_includes_llm_status(store):
    import server as _srv
    stats = _srv.Recall(store).stats()
    assert "v6_llm" in stats
    assert stats["v6_llm"]["llm_enabled"] is False
    assert stats["v6_llm"]["llm_mode"] == "false"
