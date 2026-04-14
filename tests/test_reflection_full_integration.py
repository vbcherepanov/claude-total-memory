"""Integration test for reflection.agent.run_full new phases (triple queue + fact merger)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture
def refl_db():
    """DB with v5 schema + 002/003 migrations + knowledge+embeddings tables."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    root = Path(__file__).parent.parent
    conn.executescript((root / "migrations" / "001_v5_schema.sql").read_text())
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT,
            project TEXT DEFAULT 'general', status TEXT DEFAULT 'open',
            summary TEXT, log_count INTEGER DEFAULT 0, branch TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, type TEXT, content TEXT, context TEXT DEFAULT '',
            project TEXT DEFAULT 'general', tags TEXT DEFAULT '[]',
            status TEXT DEFAULT 'active', confidence REAL DEFAULT 1.0,
            recall_count INTEGER DEFAULT 0, last_recalled TEXT,
            last_confirmed TEXT, superseded_by INTEGER, source TEXT DEFAULT 'explicit',
            created_at TEXT, updated_at TEXT, branch TEXT DEFAULT ''
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
            content, context, tags, content='knowledge', content_rowid='id'
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            knowledge_id INTEGER PRIMARY KEY,
            binary_vector BLOB NOT NULL,
            float32_vector BLOB NOT NULL,
            embed_model TEXT NOT NULL,
            embed_dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge_merges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merged_knowledge_id INTEGER NOT NULL,
            source_ids TEXT NOT NULL,
            rationale TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.executescript((root / "migrations" / "002_multi_representation.sql").read_text())
    conn.executescript((root / "migrations" / "003_triple_extraction_queue.sql").read_text())
    yield conn
    conn.close()


def test_run_full_drains_triple_queue_and_runs_fact_merger(refl_db, monkeypatch):
    """End-to-end: enqueue items → run_full → triples processed + merge attempted."""
    from reflection.agent import ReflectionAgent
    from triple_extraction_queue import TripleExtractionQueue

    # Seed knowledge + queue
    kid1 = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'User uses Go', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    kid2 = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'User prefers Go', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    refl_db.commit()
    q = TripleExtractionQueue(refl_db)
    q.enqueue(kid1)
    q.enqueue(kid2)

    # Stub ConceptExtractor.extract_and_link so it doesn't hit Ollama
    import ingestion.extractor as extractor_mod

    calls: list[int] = []

    def fake_extract_and_link(self, text, knowledge_id=None, deep=False):
        calls.append(knowledge_id)
        return {"relations": [], "entities": [], "concepts": []}

    monkeypatch.setattr(extractor_mod.ConceptExtractor, "extract_and_link", fake_extract_and_link)

    agent = ReflectionAgent(refl_db)

    # Stub fact_merger's LLM dependency: no embeddings => it returns early
    # That's fine — we just want to ensure the phase runs without error.
    report = asyncio.run(agent.run_full())

    # Triple extraction processed both queue items
    assert report["triple_extraction"]["processed"] == 2
    assert set(calls) == {kid1, kid2}

    # Fact merge phase ran (no embeddings → skipped gracefully)
    assert "fact_merge" in report
    assert report["fact_merge"].get("merged", 0) == 0  # no embeddings, no merging


def test_run_full_survives_triple_extraction_error(refl_db, monkeypatch):
    from reflection.agent import ReflectionAgent

    kid = refl_db.execute(
        "INSERT INTO knowledge (session_id, type, content, project, status, created_at) "
        "VALUES ('s1', 'fact', 'any', 'demo', 'active', '2026-04-14T00:00:00Z')"
    ).lastrowid
    refl_db.commit()
    from triple_extraction_queue import TripleExtractionQueue
    TripleExtractionQueue(refl_db).enqueue(kid)

    import ingestion.extractor as extractor_mod

    def broken(self, text, knowledge_id=None, deep=False):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(extractor_mod.ConceptExtractor, "extract_and_link", broken)

    agent = ReflectionAgent(refl_db)
    report = asyncio.run(agent.run_full())

    # Still returns a report, just with failed triples
    assert report["triple_extraction"]["failed"] >= 1
    assert report["triple_extraction"]["processed"] == 0
