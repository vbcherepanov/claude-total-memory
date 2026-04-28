"""v11.0 Phase 6 + 6b + 7 — embedding cache, space-filtered search, new MCP tools.

Three deliverables of this suite:

1. `embedding_cache_v11` table works: put -> get roundtrip, hit_count
   increments on every retrieval.

2. `Recall.search(embedding_space=...)` filters the candidate pool to
   rows whose `embeddings.embedding_space` matches. This subsumes the
   xfail `test_search_filter_by_embedding_space` from the Phase 2 suite.

3. The new MCP tools (`memory_explain_search`, `memory_warmup`) return
   the documented payload shape.

Tests reuse the `fast_store` fixture pattern from `test_no_llm_hot_path_v11`
(production-default env, FastEmbed available, no LLM allowed). They do not
import `llm_provider` themselves.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


SRC = str(Path(__file__).parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ──────────────────────────────────────────────
# Fast store fixture (FastEmbed-only, no LLM).
# ──────────────────────────────────────────────


@pytest.fixture
def fast_store(monkeypatch, tmp_path):
    """A Store running in v11 fast mode with FastEmbed available."""
    for var in (
        "MEMORY_QUALITY_GATE_ENABLED",
        "MEMORY_CONTRADICTION_DETECT_ENABLED",
        "MEMORY_ENTITY_DEDUP_ENABLED",
        "MEMORY_COREF_ENABLED",
        "USE_ADVANCED_RAG",
        "MEMORY_QUERY_REWRITE",
        "MEMORY_ASYNC_ENRICHMENT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MEMORY_MODE", "fast")
    monkeypatch.setenv("MEMORY_USE_LLM_IN_HOT_PATH", "false")
    monkeypatch.setenv("MEMORY_ALLOW_OLLAMA_IN_HOT_PATH", "false")
    monkeypatch.setenv("MEMORY_RERANK_ENABLED", "false")
    monkeypatch.setenv("MEMORY_ENRICHMENT_ENABLED", "false")
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")

    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server, config

    if hasattr(config, "_cache_clear"):
        config._cache_clear()
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()
    s.db.execute(
        "INSERT INTO sessions (id, started_at, project, status) "
        "VALUES ('s1','2026-04-27T00:00:00Z','demo','open')"
    )
    s.db.commit()
    yield s
    try:
        s.db.close()
    except Exception:
        pass


# ──────────────────────────────────────────────
# (A) Phase 7 — embedding cache roundtrip
# ──────────────────────────────────────────────


def test_embedding_cache_put_get_roundtrip(fast_store):
    """Put a vector, retrieve it via cache_key, hit_count increments."""
    from memory_core import embedding_cache as ec

    # Migration 022 ran at Store init.
    n_table = fast_store.db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='embedding_cache_v11'"
    ).fetchone()
    assert n_table is not None, "migration 022 did not create embedding_cache_v11"

    key = ec.cache_key(
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        space="text",
        normalized_content="hello world",
    )
    vec = [0.1, 0.2, 0.3, 0.4]

    miss = ec.get(fast_store.db, key)
    assert miss is None, "fresh cache must miss"

    ec.put(
        fast_store.db, key, vec,
        provider="fastembed",
        model="BAAI/bge-small-en-v1.5",
        space="text",
    )

    got = ec.get(fast_store.db, key)
    assert got is not None
    # struct.pack/unpack 'f' loses precision; use approx.
    assert len(got) == len(vec)
    for a, b in zip(got, vec):
        assert abs(a - b) < 1e-6

    # Two more reads → hit_count must be 3.
    ec.get(fast_store.db, key)
    ec.get(fast_store.db, key)
    row = fast_store.db.execute(
        "SELECT hit_count FROM embedding_cache_v11 WHERE cache_key=?",
        (key,),
    ).fetchone()
    assert row is not None
    assert int(row[0]) == 3, f"expected hit_count=3, got {row[0]}"

    # stats() reflects the activity.
    st = ec.stats(fast_store.db)
    assert st["rows"] == 1
    assert st["total_hits"] == 3
    assert st["by_space"].get("text") == 1


def test_embedding_cache_lru_vacuum(fast_store):
    """`vacuum(max_rows=N)` evicts the oldest rows by `last_used_at`."""
    from memory_core import embedding_cache as ec

    for i in range(5):
        ec.put(
            fast_store.db,
            ec.cache_key("fastembed", "m", "text", f"text-{i}"),
            [float(i)] * 4,
            provider="fastembed",
            model="m",
            space="text",
        )

    deleted = ec.vacuum(fast_store.db, max_rows=2)
    assert deleted == 3, f"expected 3 evictions, got {deleted}"
    rows = fast_store.db.execute(
        "SELECT COUNT(*) FROM embedding_cache_v11"
    ).fetchone()[0]
    assert rows == 2


# ──────────────────────────────────────────────
# (B) Phase 6b — embedding_space filter
# ──────────────────────────────────────────────


def test_search_filter_by_embedding_space_real(fast_store):
    """Save a text-typed and a code-typed memory, search with
    embedding_space='code' → only the code one comes back, and every
    returned row exposes embedding_space='code' in detail='full' mode.
    """
    text_rid, *_ = fast_store.save_knowledge(
        sid="s1",
        content="postgres autovacuum threshold tuning notes — set to 10% of dead tuples",
        ktype="fact",
        project="demo",
    )
    code_rid, *_ = fast_store.save_knowledge(
        sid="s1",
        content=(
            "import statistics\n"
            "from typing import Iterable\n"
            "\n"
            "def calculate_median(values: list[float]) -> float:\n"
            "    sorted_v = sorted(values)\n"
            "    n = len(sorted_v)\n"
            "    return sorted_v[n // 2]\n"
            "\n"
            "async def median_async(it: Iterable[float]) -> float:\n"
            "    return calculate_median(list(it))\n"
        ),
        ktype="solution",
        project="demo",
    )
    assert text_rid != code_rid

    # Sanity: classifier put them in different spaces.
    spaces = dict(fast_store.db.execute(
        "SELECT knowledge_id, embedding_space FROM embeddings "
        "WHERE knowledge_id IN (?, ?)", (text_rid, code_rid),
    ).fetchall())
    assert spaces[text_rid] == "text"
    assert spaces[code_rid] == "code"

    import server as _srv
    recall = _srv.Recall(fast_store)

    # No filter → both eligible (FTS may surface only the matching one,
    # that's fine — we only assert the filter narrows the pool).
    result_all = recall.search(query="median", project="demo", limit=10)
    all_ids = {
        item["id"]
        for grp in result_all["results"].values()
        for item in grp
    }
    # The text fact doesn't mention 'median'; it should not surface even
    # without a filter, but the test keeps this loose so behaviour drift
    # in BM25 doesn't break it.

    # With embedding_space='code' → ONLY code rows allowed; text fact is
    # excluded even when its content matched (it doesn't here, but the
    # contract holds).
    result_code = recall.search(
        query="median", project="demo", limit=10,
        embedding_space="code",
    )
    code_items = [
        item
        for grp in result_code["results"].values()
        for item in grp
    ]
    assert code_items, "code-space search returned nothing"
    for it in code_items:
        assert it.get("embedding_space") == "code", (
            f"row id={it['id']} surfaced under embedding_space='code' filter "
            f"but row says embedding_space={it.get('embedding_space')!r}"
        )
    # The text record must not appear.
    assert text_rid not in {it["id"] for it in code_items}

    # Top-level result echoes the requested space.
    assert result_code.get("embedding_space") == "code"


def test_search_embedding_space_list(fast_store):
    """`embedding_space=[...]` accepts multiple spaces."""
    rid_text, *_ = fast_store.save_knowledge(
        sid="s1",
        content="redis cluster failover playbook — promote replica then re-shard",
        ktype="fact", project="demo",
    )
    rid_code, *_ = fast_store.save_knowledge(
        sid="s1",
        content="def failover(cluster): cluster.promote_replica()",
        ktype="solution", project="demo",
    )

    import server as _srv
    recall = _srv.Recall(fast_store)
    result = recall.search(
        query="failover", project="demo", limit=10,
        embedding_space=["text", "code"],
    )
    ids = {item["id"] for grp in result["results"].values() for item in grp}
    # Both rows are in the {text, code} union — both eligible.
    assert rid_text in ids or rid_code in ids


# ──────────────────────────────────────────────
# (C) Phase 6 — new MCP tools
# ──────────────────────────────────────────────


def _call_tool(name: str, args: dict) -> dict:
    """Invoke `_do(name, args)` against the module-level `store`/`recall`."""
    import server

    server.store = server.Store() if not hasattr(server, "store") else server.store
    if not hasattr(server, "recall") or server.recall is None:
        server.recall = server.Recall(server.store)
    if not hasattr(server, "SID") or server.SID is None:
        server.SID = "test-session"
    if not hasattr(server, "BRANCH"):
        server.BRANCH = ""
    raw = asyncio.run(server._do(name, args))
    return json.loads(raw)


@pytest.fixture
def wired_server(fast_store, monkeypatch):
    """Wire the module-level globals that `_do()` reads."""
    import server
    monkeypatch.setattr(server, "store", fast_store)
    monkeypatch.setattr(server, "recall", server.Recall(fast_store))
    monkeypatch.setattr(server, "SID", "s1")
    monkeypatch.setattr(server, "BRANCH", "", raising=False)
    return server


def test_memory_explain_search_returns_breakdown(wired_server):
    """memory_explain_search must include {fts, semantic, graph, merged,
    embedding_space} keys in `_explain`."""
    server = wired_server
    server.store.save_knowledge(
        sid="s1",
        content="kubernetes operator pattern with custom resource definitions",
        ktype="fact", project="demo",
    )
    out = asyncio.run(server._do("memory_explain_search", {
        "query": "kubernetes operator", "project": "demo", "limit": 5,
    }))
    payload = json.loads(out)
    assert "_explain" in payload, payload
    explain = payload["_explain"]
    for key in ("fts", "semantic", "graph", "merged", "embedding_space"):
        assert key in explain, f"_explain is missing key {key!r}: {explain}"
    assert isinstance(explain["fts"], list)
    assert isinstance(explain["semantic"], list)
    assert isinstance(explain["graph"], list)
    assert isinstance(explain["merged"], list)
    # rerank_applied must be False in fast mode.
    assert explain.get("rerank_applied") is False


def test_memory_warmup_returns_status(wired_server):
    """memory_warmup must report fastembed_loaded=True after running."""
    server = wired_server
    out = asyncio.run(server._do("memory_warmup", {}))
    payload = json.loads(out)
    assert "fastembed_loaded" in payload
    assert "vector_backend" in payload
    assert "ms" in payload
    # In the test environment FastEmbed is available; either it's loaded
    # via fastembed or via the SentenceTransformer fallback. Both are
    # acceptable — the contract is just "an embedder is available".
    assert payload["fastembed_loaded"] is True, payload


def test_memory_save_fast_skips_quality_gate(wired_server, monkeypatch):
    """memory_save_fast must succeed even when QUALITY_GATE is on, because
    it forces skip_quality=True."""
    monkeypatch.setenv("MEMORY_QUALITY_GATE_ENABLED", "true")
    server = wired_server
    out = asyncio.run(server._do("memory_save_fast", {
        "content": "x",  # would normally be too short for the gate
        "type": "fact",
        "project": "demo",
    }))
    payload = json.loads(out)
    assert payload.get("saved") is True, payload
    assert payload.get("mode") == "fast"
    assert isinstance(payload.get("id"), int)


def test_memory_search_fast_returns_results(wired_server):
    server = wired_server
    server.store.save_knowledge(
        sid="s1",
        content="redis cluster failover scenarios — promote replica via CLUSTER FAILOVER",
        ktype="fact", project="demo",
    )
    out = asyncio.run(server._do("memory_search_fast", {
        "query": "redis cluster", "project": "demo", "limit": 5,
    }))
    payload = json.loads(out)
    assert payload.get("mode") == "fast"
    total = payload.get("total", 0)
    assert total >= 1


def test_memory_perf_report_has_counters(wired_server):
    server = wired_server
    # Run something so counters move.
    server.store.save_knowledge(
        sid="s1", content="sample content", ktype="fact", project="demo",
    )
    asyncio.run(server._do("memory_search_fast", {
        "query": "sample", "project": "demo",
    }))
    out = asyncio.run(server._do("memory_perf_report", {}))
    payload = json.loads(out)
    assert "counters" in payload
    counters = payload["counters"]
    # llm_calls / network_calls must stay 0 on the fast hot path.
    assert counters.get("llm_calls", 0) == 0
    assert counters.get("network_calls", 0) == 0
    # search_total_ms should have moved.
    assert "search_total_ms" in counters
    # embedding_cache stats present.
    assert "embedding_cache_v11" in payload


def test_memory_rebuild_fts_idempotent(wired_server):
    server = wired_server
    server.store.save_knowledge(
        sid="s1", content="postgres vacuum", ktype="fact", project="demo",
    )
    out = asyncio.run(server._do("memory_rebuild_fts", {}))
    payload = json.loads(out)
    assert payload.get("rebuilt") is True
    assert payload["rows_after"] >= 1


def test_memory_rebuild_embeddings_reencode(wired_server):
    server = wired_server
    rid, *_ = server.store.save_knowledge(
        sid="s1",
        content=(
            "import math\n"
            "from typing import Sequence\n"
            "\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            "async def add_async(s: Sequence[int]) -> int:\n"
            "    return sum(s)\n"
        ),
        ktype="solution", project="demo",
    )
    # Drop the existing row to force a re-encode.
    server.store.db.execute(
        "UPDATE embeddings SET embedding_space='text' WHERE knowledge_id=?",
        (rid,),
    )
    server.store.db.commit()

    out = asyncio.run(server._do("memory_rebuild_embeddings", {
        "embedding_space": "text",
        "project": "demo",
    }))
    payload = json.loads(out)
    assert payload.get("rebuilt", 0) >= 1

    # After re-encode, the code chunk should be back in the code space.
    space = server.store.db.execute(
        "SELECT embedding_space FROM embeddings WHERE knowledge_id=?",
        (rid,),
    ).fetchone()[0]
    assert space == "code", f"expected code, got {space!r}"
