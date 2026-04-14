"""Ensure rerank=True works when multi_repr tier contributes records to the pool."""

from __future__ import annotations

import asyncio
import hashlib
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


def test_rerank_includes_multi_repr_hits(store, monkeypatch):
    """Records surfaced via multi_repr must participate in rerank pool without crashing."""
    import representations, deep_enricher
    import ingestion.extractor as extractor_mod
    from reflection.agent import ReflectionAgent

    store.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('s1', '2026-04-14T00:00:00Z', 'demo', 'open')"
    )
    store.db.commit()

    long_content = (
        "A production playbook for event-driven architecture using RabbitMQ 4, "
        "covering message routing, dead-letter queues, quorum queues, delayed "
        "delivery, and high-availability patterns. Consumers use manual ack "
        "with prefetch tuning for optimal throughput and back-pressure handling."
    )
    rid, _, _ = store.save_knowledge(
        sid="s1", content=long_content, ktype="solution", project="demo"
    )

    def fake_llm(prompt, **_):
        p = prompt.lower()
        if "summary" in p or "summarize" in p:
            return "RabbitMQ 4 playbook for event-driven patterns and HA."
        if "keyword" in p:
            return "rabbitmq, event-driven, quorum, dlq"
        if "question" in p:
            return "How to set up dead-letter queues?\nWhat is quorum queue?"
        return ""

    monkeypatch.setattr(representations, "_llm_complete", fake_llm)
    monkeypatch.setattr(deep_enricher, "_llm_complete", lambda *a, **kw: "")
    monkeypatch.setattr(
        extractor_mod.ConceptExtractor, "extract_and_link",
        lambda self, text, knowledge_id=None, deep=False: {"relations": []},
    )

    def emb(text: str) -> list[float]:
        embs = store.embed([text])
        return embs[0] if embs else []

    asyncio.run(ReflectionAgent(store.db, embedder=emb).run_full())

    import server as _srv
    recall = _srv.Recall(store)

    # Rerank enabled — must not crash even if CE is unavailable locally
    result = recall.search(
        query="event-driven messaging quorum",
        project="demo",
        limit=10,
        rerank=True,
    )

    all_items = []
    for group in result.get("results", {}).values():
        all_items.extend(group)
    assert any(item.get("id") == rid for item in all_items), "record missing from results"
