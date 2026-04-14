"""Test the fast `run_drain` scope — drains queues without digest/synthesize."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture
def drain_db():
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
            content TEXT, type TEXT, project TEXT DEFAULT 'general',
            tags TEXT DEFAULT '[]', status TEXT DEFAULT 'active',
            confidence REAL DEFAULT 1.0, created_at TEXT, session_id TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            knowledge_id INTEGER PRIMARY KEY,
            binary_vector BLOB NOT NULL, float32_vector BLOB NOT NULL,
            embed_model TEXT NOT NULL, embed_dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge_merges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merged_knowledge_id INTEGER NOT NULL,
            source_ids TEXT NOT NULL, rationale TEXT, created_at TEXT NOT NULL
        );
        """
    )
    for m in ("002_multi_representation", "003_triple_extraction_queue",
              "004_deep_enrichment", "005_representations_queue"):
        conn.executescript((root / "migrations" / f"{m}.sql").read_text())
    yield conn
    conn.close()


def test_run_drain_only_runs_phases_3_5_6(drain_db, monkeypatch):
    from reflection.agent import ReflectionAgent
    from triple_extraction_queue import TripleExtractionQueue
    from deep_enrichment_queue import DeepEnrichmentQueue
    from representations_queue import RepresentationsQueue

    kid = drain_db.execute(
        "INSERT INTO knowledge (content, type, project, created_at) "
        "VALUES ('quick test content', 'fact', 'demo', '2026-04-14T00:00:00Z')"
    ).lastrowid
    drain_db.commit()
    TripleExtractionQueue(drain_db).enqueue(kid)
    DeepEnrichmentQueue(drain_db).enqueue(kid)
    RepresentationsQueue(drain_db).enqueue(kid)

    # Stub heavy dependencies so no Ollama/FastEmbed involvement
    import ingestion.extractor as extractor_mod
    import deep_enricher, representations

    monkeypatch.setattr(
        extractor_mod.ConceptExtractor, "extract_and_link",
        lambda self, text, knowledge_id=None, deep=False: {"relations": []},
    )
    monkeypatch.setattr(deep_enricher, "_llm_complete", lambda *a, **kw: "")
    monkeypatch.setattr(representations, "_llm_complete", lambda *a, **kw: "")

    # Digest/synthesize classes must NOT be invoked — sentinel any call
    from reflection.digest import DigestPhase
    from reflection.synthesize import SynthesizePhase
    digest_calls, synth_calls = [], []
    monkeypatch.setattr(DigestPhase, "run", lambda self: digest_calls.append(1) or {})
    monkeypatch.setattr(SynthesizePhase, "run", lambda self, days=7: synth_calls.append(1) or {})

    agent = ReflectionAgent(drain_db, embedder=lambda t: [0.1] * 8)
    report = asyncio.run(agent.run_drain())

    assert report["scope"] == "drain"
    assert "digest" not in report
    assert "synthesis" not in report
    assert report["triple_extraction"]["processed"] == 1
    assert report["deep_enrichment"]["processed"] == 1
    assert report["representations"]["processed"] == 1
    assert digest_calls == []
    assert synth_calls == []


def test_pick_scope_auto_prefers_drain():
    from tools.run_reflection import pick_scope

    assert pick_scope({"a": 0, "b": 0, "c": 0}) == "quick"
    assert pick_scope({"a": 1, "b": 1, "c": 1}) == "drain"
    assert pick_scope({"a": 100, "b": 100, "c": 100}) == "drain"  # always drain
    assert pick_scope({"a": 0}, override="full") == "full"
    assert pick_scope({"a": 99}, override="quick") == "quick"
