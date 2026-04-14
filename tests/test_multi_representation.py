"""Tests for multi-representation embeddings (GEM-RAG)."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def repr_db():
    """SQLite in-memory DB with v5 schema + migration 002."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    root = Path(__file__).parent.parent
    conn.executescript((root / "migrations" / "001_v5_schema.sql").read_text())
    # base tables
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT, project TEXT DEFAULT 'general',
            status TEXT DEFAULT 'active', created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            knowledge_id INTEGER PRIMARY KEY,
            binary_vector BLOB NOT NULL,
            float32_vector BLOB NOT NULL,
            embed_model TEXT NOT NULL,
            embed_dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.executescript((root / "migrations" / "002_multi_representation.sql").read_text())
    yield conn
    conn.close()


def _fake_embed(text: str, dim: int = 8) -> list[float]:
    """Deterministic fake embedding from text (for unit tests)."""
    import hashlib

    h = hashlib.md5(text.encode()).digest()
    return [b / 255.0 for b in h[:dim]]


# ──────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────


def test_store_upsert_and_get(repr_db):
    from multi_repr_store import MultiReprStore

    s = MultiReprStore(repr_db)
    kid = repr_db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, ?)",
        ("hello world", "2026-01-01T00:00:00Z"),
    ).lastrowid

    s.upsert(kid, representation="summary", content="hi", embedding=_fake_embed("hi"), model="test")
    s.upsert(
        kid, representation="keywords", content="hello,world",
        embedding=_fake_embed("hello,world"), model="test",
    )

    reps = s.get_all_for(kid)
    assert {r["representation"] for r in reps} == {"summary", "keywords"}


def test_store_replaces_on_re_upsert(repr_db):
    from multi_repr_store import MultiReprStore

    s = MultiReprStore(repr_db)
    kid = repr_db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, ?)",
        ("x", "2026-01-01T00:00:00Z"),
    ).lastrowid

    s.upsert(kid, representation="summary", content="first", embedding=_fake_embed("first"), model="test")
    s.upsert(kid, representation="summary", content="second", embedding=_fake_embed("second"), model="test")

    rows = repr_db.execute(
        "SELECT content FROM knowledge_representations WHERE knowledge_id=? AND representation='summary'",
        (kid,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["content"] == "second"


def test_store_delete_cascades_on_representation(repr_db):
    from multi_repr_store import MultiReprStore

    s = MultiReprStore(repr_db)
    kid = repr_db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, ?)",
        ("x", "2026-01-01T00:00:00Z"),
    ).lastrowid
    s.upsert(kid, "summary", "a", _fake_embed("a"), "test")
    s.upsert(kid, "keywords", "b", _fake_embed("b"), "test")

    s.delete_all_for(kid)

    cnt = repr_db.execute(
        "SELECT COUNT(*) FROM knowledge_representations WHERE knowledge_id=?", (kid,)
    ).fetchone()[0]
    assert cnt == 0


def test_store_rejects_invalid_type(repr_db):
    from multi_repr_store import MultiReprStore

    s = MultiReprStore(repr_db)
    kid = repr_db.execute(
        "INSERT INTO knowledge (content, created_at) VALUES (?, ?)",
        ("x", "2026-01-01T00:00:00Z"),
    ).lastrowid
    with pytest.raises(ValueError):
        s.upsert(kid, "bogus", "x", _fake_embed("x"), "test")


# ──────────────────────────────────────────────
# Representation generators (stub LLM)
# ──────────────────────────────────────────────


def test_generators_return_expected_shape(monkeypatch):
    from representations import generate_representations

    # Stub the LLM call
    def fake_llm(prompt: str, **_: object) -> str:
        if "summarize" in prompt.lower() or "summary" in prompt.lower():
            return "Short summary of content."
        if "keyword" in prompt.lower():
            return "go, backend, api, microservice"
        if "question" in prompt.lower():
            return "What is this?\nHow does it work?\nWhen to use?"
        return ""

    monkeypatch.setattr("representations._llm_complete", fake_llm)

    long_text = (
        "A long document about Go backend microservices using gRPC. "
        "The system uses event-driven architecture with RabbitMQ for messaging. "
        "Data persisted in PostgreSQL 18 with UUID v7 primary keys. "
        "Services deployed via Docker Compose with health checks and readiness probes. "
        "Observability: Prometheus metrics, Grafana dashboards, Loki for log aggregation. "
    )
    result = generate_representations(long_text, project="demo")

    assert set(result) >= {"summary", "keywords", "questions"}
    assert result["summary"].strip()
    # Keywords: list-like
    assert "," in result["keywords"] or "\n" in result["keywords"]
    # Questions: multiple lines
    assert result["questions"].count("\n") >= 1 or "?" in result["questions"]


def test_generators_skip_on_short_text(monkeypatch):
    """Short text (<50 tokens) shouldn't waste LLM calls on summary."""
    from representations import generate_representations

    calls: list[str] = []

    def fake_llm(prompt: str, **_: object) -> str:
        calls.append(prompt)
        return "x"

    monkeypatch.setattr("representations._llm_complete", fake_llm)

    result = generate_representations("tiny text", project="demo")

    # Summary should be skipped (too short); keywords/questions may still run
    assert "summary" not in result or result.get("summary") == ""


def test_generators_graceful_llm_failure(monkeypatch):
    """If LLM is down, return whatever succeeded — don't crash."""
    from representations import generate_representations

    def broken_llm(prompt: str, **_: object) -> str:
        raise RuntimeError("ollama down")

    monkeypatch.setattr("representations._llm_complete", broken_llm)

    result = generate_representations("some longer text " * 30, project="demo")

    # No representations generated but no exception raised
    assert isinstance(result, dict)
    for v in result.values():
        assert v == "" or v is None


# ──────────────────────────────────────────────
# Search fusion
# ──────────────────────────────────────────────


def test_rrf_fusion_combines_ranks():
    from multi_repr_store import rrf_fuse

    # Three ranked lists (representation → [(knowledge_id, score), ...])
    ranked = {
        "raw":      [(1, 0.9), (2, 0.8), (3, 0.7)],
        "summary":  [(2, 0.95), (4, 0.8), (1, 0.6)],
        "keywords": [(3, 0.9), (1, 0.85)],
    }
    fused = rrf_fuse(ranked, k=60, top_n=4)

    # id=1 appears in all three → should be among top
    ids = [kid for kid, _score in fused]
    assert 1 in ids[:2]
    assert len(fused) <= 4
    # Fused score is float-sorted descending
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fusion_empty_inputs():
    from multi_repr_store import rrf_fuse

    assert rrf_fuse({}, k=60, top_n=5) == []
    assert rrf_fuse({"raw": []}, k=60, top_n=5) == []
