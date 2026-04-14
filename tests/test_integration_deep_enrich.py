"""Integration: memory_save → deep_enrichment_queue; reflection drains it."""

from __future__ import annotations

import asyncio
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


def test_save_enqueues_deep_enrichment(store):
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()

    rid, _, _ = store.save_knowledge(
        sid="s1", content="content to enrich", ktype="fact", project="demo"
    )

    row = store.db.execute(
        "SELECT status FROM deep_enrichment_queue WHERE knowledge_id=?", (rid,)
    ).fetchone()
    assert row is not None
    assert row["status"] == "pending"


def test_reflection_drains_deep_enrichment(store, monkeypatch):
    from reflection.agent import ReflectionAgent
    import deep_enricher

    # Seed session + knowledge + enqueue (happens automatically via save)
    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()
    # Content must exceed MIN_CHARS_FOR_LLM (120) so deep_enrich calls the LLM
    long_content = (
        "A thorough document about Go microservices built with gRPC transport. "
        "Auth flow uses JWT tokens verified against a PostgreSQL 18 backing store. "
        "Services are deployed via Docker Compose with health checks, metrics, "
        "and Loki log aggregation for observability."
    )
    rid, _, _ = store.save_knowledge(
        sid="s1", content=long_content, ktype="fact", project="demo"
    )

    # Stub LLM in deep_enricher so we don't hit Ollama
    def fake_llm(prompt, **_):
        if "entit" in prompt.lower():
            return '{"entities":[{"name":"Go","type":"technology"},{"name":"PostgreSQL","type":"technology"}]}'
        if "intent" in prompt.lower():
            return "procedural"
        if "topic" in prompt.lower():
            return '["microservices","auth","db"]'
        return ""

    monkeypatch.setattr(deep_enricher, "_llm_complete", fake_llm)

    # Run reflection — drains the enrichment queue
    agent = ReflectionAgent(store.db)
    # Also stub ConceptExtractor to avoid Ollama in triple phase
    import ingestion.extractor as extractor_mod
    monkeypatch.setattr(
        extractor_mod.ConceptExtractor, "extract_and_link",
        lambda self, text, knowledge_id=None, deep=False: {"relations": [], "entities": []},
    )

    report = asyncio.run(agent.run_full())

    assert report["deep_enrichment"]["processed"] >= 1

    row = store.db.execute(
        "SELECT entities, intent, topics FROM knowledge_enrichment WHERE knowledge_id=?",
        (rid,),
    ).fetchone()
    assert row is not None
    assert row["intent"] == "procedural"
    assert "microservices" in row["topics"]
    assert "Go" in row["entities"]
