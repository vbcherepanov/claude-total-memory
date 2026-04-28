"""v11.0 Phase 2 regression — fast mode must do zero LLM / network calls.

These tests are written BEFORE Phase 1 flips the defaults. On v10.5.0 they
fail (because `quality_gate` and `contradiction_detector` synchronously call
the LLM during `save_knowledge`). After Phase 1 introduces
`MEMORY_MODE=fast` and short-circuits A1/A5/A6/A13/A14 + B1/B2/B3, these
tests turn green and lock in the new contract.

The tests intentionally do NOT use the conftest fixtures that disable the
quality gate / contradiction detector. We want the *production default*
behaviour to be measured; that is what production users get.

How "no LLM" is enforced:

1. `llm_provider.make_provider` is monkeypatched to return a provider whose
   `.complete()` increments a counter and raises. The counter must stay 0.
2. `socket.create_connection` is monkeypatched to fail loudly so any
   accidental HTTP/Ollama call surfaces as a clear test failure rather than
   a silent timeout.
3. `urllib.request.urlopen` is also patched.

Net effect: any sync LLM-touching code path on the save/search hot path
becomes immediately visible.
"""

from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path

import pytest


SRC = str(Path(__file__).parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


# ──────────────────────────────────────────────
# Trip-wire counters
# ──────────────────────────────────────────────


class HotPathTripwire:
    """Collects any LLM / network attempts made during a test."""

    def __init__(self) -> None:
        self.llm_calls: list[str] = []
        self.network_calls: list[str] = []

    def record_llm(self, where: str) -> None:
        self.llm_calls.append(where)

    def record_network(self, target: str) -> None:
        self.network_calls.append(target)

    def assert_zero(self) -> None:
        assert self.llm_calls == [], (
            f"hot path made {len(self.llm_calls)} LLM call(s): {self.llm_calls}"
        )
        assert self.network_calls == [], (
            f"hot path made {len(self.network_calls)} network call(s): "
            f"{self.network_calls}"
        )


@pytest.fixture
def tripwire(monkeypatch):
    """Install LLM + network tripwires before any Store operation."""
    tw = HotPathTripwire()

    # 1. Block llm_provider.make_provider — anything touching LLM must hit this.
    import llm_provider

    class _BlockingProvider:
        name = "blocking"

        def available(self) -> bool:
            return True

        def complete(self, prompt, **kwargs):  # noqa: ANN001, ANN003
            tw.record_llm(f"llm_provider.complete(prompt[:40]={prompt[:40]!r})")
            raise RuntimeError(
                "v11.0 fast hot-path tripwire: LLM call attempted during "
                "save/search. This indicates a regression — the fast path "
                "must be 100% deterministic."
            )

    monkeypatch.setattr(
        llm_provider, "make_provider", lambda *_a, **_kw: _BlockingProvider()
    )

    # 2. Block any direct HTTP attempt (Ollama, anthropic, openai).
    real_create_connection = socket.create_connection

    def _blocked_socket(addr, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        host, port = addr
        # Localhost SQLite/server traffic on a random ephemeral port is fine
        # only if it isn't 11434 (Ollama default). Anything else is a tripwire.
        if host in ("127.0.0.1", "localhost") and port in (
            11434,  # Ollama
        ):
            tw.record_network(f"{host}:{port}")
            raise OSError(
                "v11.0 fast hot-path tripwire: network call to "
                f"{host}:{port} blocked"
            )
        # Real internet IP / DNS attempts? Block.
        if not host.startswith(("127.", "::1", "localhost")):
            tw.record_network(f"{host}:{port}")
            raise OSError(
                "v11.0 fast hot-path tripwire: external network call "
                f"to {host}:{port} blocked"
            )
        return real_create_connection(addr, *a, **kw)

    monkeypatch.setattr(socket, "create_connection", _blocked_socket)

    real_urlopen = urllib.request.urlopen

    def _blocked_urlopen(req, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        url = getattr(req, "full_url", str(req))
        tw.record_network(f"urlopen:{url}")
        raise OSError(
            f"v11.0 fast hot-path tripwire: urlopen({url}) blocked"
        )

    monkeypatch.setattr(urllib.request, "urlopen", _blocked_urlopen)

    return tw


@pytest.fixture
def fast_store(monkeypatch, tmp_path, tripwire):
    """A Store running in v11 fast mode. Phase 1 will read MEMORY_MODE."""
    # Production-shaped env: clear knobs that conftest set, then enable fast.
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
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")  # belt + braces

    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    import server, config

    if hasattr(config, "_cache_clear"):
        config._cache_clear()
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    s = server.Store()

    # Seed a session row that save_knowledge expects.
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
# Save hot-path tests
# ──────────────────────────────────────────────


def test_save_fact_makes_no_llm_calls(fast_store, tripwire):
    """Saving a low-stakes fact must not touch the LLM in fast mode."""
    rid, dup, *_ = fast_store.save_knowledge(
        sid="s1",
        content="postgres pg_upgrade requires matching collation across major versions",
        ktype="fact",
        project="demo",
    )
    assert rid is not None
    tripwire.assert_zero()


def test_save_solution_makes_no_llm_calls(fast_store, tripwire):
    """`solution`/`decision` types historically triggered the contradiction
    detector synchronously. Fast mode must skip it."""
    rid, dup, *_ = fast_store.save_knowledge(
        sid="s1",
        content="fix: bump pgbouncer pool_size from 25 to 100 to clear connection storms",
        ktype="solution",
        project="demo",
    )
    assert rid is not None
    tripwire.assert_zero()


def test_save_decision_makes_no_llm_calls(fast_store, tripwire):
    rid, dup, *_ = fast_store.save_knowledge(
        sid="s1",
        content="decision: keep ChromaDB as the vector backend in v11 — wrap in memory_core/vector_store",
        ktype="decision",
        project="demo",
        context="WHY: HNSW + binary quantization already proven at 97.4% R@5",
    )
    assert rid is not None
    tripwire.assert_zero()


# ──────────────────────────────────────────────
# Search / recall hot-path tests
# ──────────────────────────────────────────────


def test_search_makes_no_llm_calls(fast_store, tripwire):
    """HyDE / analyze_query / query_rewriter must not fire in fast search."""
    fast_store.save_knowledge(
        sid="s1",
        content="kubernetes operator pattern with custom resource definitions",
        ktype="fact",
        project="demo",
    )
    # The save call already verified zero LLM, but the search may have its own.
    tripwire.llm_calls.clear()
    tripwire.network_calls.clear()

    import server as _srv

    recall = _srv.Recall(fast_store)
    result = recall.search(query="kubernetes operator", project="demo", limit=5)
    assert result is not None
    tripwire.assert_zero()


def test_repeated_search_makes_no_llm_calls(fast_store, tripwire):
    fast_store.save_knowledge(
        sid="s1", content="redis cluster failover scenarios", ktype="fact", project="demo"
    )
    tripwire.llm_calls.clear()
    tripwire.network_calls.clear()

    import server as _srv

    recall = _srv.Recall(fast_store)
    for _ in range(3):
        recall.search(query="redis cluster", project="demo", limit=5)
    tripwire.assert_zero()


# ──────────────────────────────────────────────
# Embed silent fallback test
# ──────────────────────────────────────────────


def test_embed_does_not_silently_fall_back_to_ollama(fast_store, monkeypatch, tripwire):
    """When FastEmbed fails and `MEMORY_ALLOW_OLLAMA_IN_HOT_PATH=false`,
    `Store.embed` must raise (or return []) instead of probing Ollama.

    On v10.5.0 this test fails: `embed()` falls through to `_ollama_embed`,
    which calls `urlopen(http://localhost:11434/api/tags)`. The tripwire
    catches that as a network call. After Phase 1 the ladder is gated and
    no Ollama probe happens.
    """
    # Force the FastEmbed and SentenceTransformer properties to look broken.
    # Both are @property on Store, so we replace them on the instance via
    # the underlying caches, not via direct attribute assignment.
    import server as _srv

    monkeypatch.setattr(
        type(fast_store), "fastembed", property(lambda self: None)
    )
    monkeypatch.setattr(
        type(fast_store), "embedder", property(lambda self: None)
    )
    # Ensure cached Ollama state doesn't short-circuit either way.
    fast_store._ollama_available = None
    # Make _fastembed_embed return [] so the ladder is forced to consider
    # the next rung (Ollama). On v10.5 it does; in v11 fast it must not.
    monkeypatch.setattr(fast_store, "_fastembed_embed", lambda batch: [])

    # If the system silently probes Ollama, the tripwire records a network
    # call. If it raises cleanly without a probe, the tripwire stays empty.
    try:
        result = fast_store.embed(["any text"])
    except Exception:
        result = None  # raising is also acceptable in fast mode

    # The contract: zero Ollama probes, regardless of whether embed() raised
    # or returned []/None.
    tripwire.assert_zero()
    # And the result must NOT be a non-empty Ollama-derived vector list.
    assert not result, (
        "Fast mode produced an embedding via fallback even though FastEmbed "
        "was disabled — silent fallback is forbidden."
    )


# ──────────────────────────────────────────────
# Module-level import guards (for Phase 3/4 — currently xfail)
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Multi-embedding-space contract (Phase 1b / 6b)
# ──────────────────────────────────────────────


def test_save_writes_embedding_space_metadata(fast_store):
    """Every embedding row must carry an embedding_space column."""
    rid, *_ = fast_store.save_knowledge(
        sid="s1", content="postgres autovacuum threshold tuning notes",
        ktype="fact", project="demo",
    )
    row = fast_store.db.execute(
        "SELECT embedding_space, embedding_provider, content_type "
        "FROM embeddings WHERE knowledge_id=?",
        (rid,),
    ).fetchone()
    assert row is not None, "embedding row missing"
    assert row[0] in ("text", "markdown", "code", "log", "config"), (
        f"embedding_space must be set, got {row[0]!r}"
    )


def test_existing_vectors_get_text_space_after_migration(fast_store):
    """The 021 backfill UPDATE must populate every pre-v11 row."""
    cnt_null = fast_store.db.execute(
        "SELECT COUNT(*) FROM embeddings WHERE embedding_space IS NULL"
    ).fetchone()[0]
    assert cnt_null == 0, (
        f"{cnt_null} legacy embedding rows still have NULL embedding_space "
        "after migration 021 — backfill missed them."
    )


@pytest.mark.xfail(
    reason="Phase 6b: Recall.search does not accept embedding_space yet.",
    strict=False,
)
def test_search_filter_by_embedding_space(fast_store):
    """Searching with embedding_space='code' must return only code rows."""
    fast_store.save_knowledge(
        sid="s1", content="postgres autovacuum threshold tuning notes",
        ktype="fact", project="demo",
    )
    fast_store.save_knowledge(
        sid="s1",
        content="def calculate_median(values: list[float]) -> float: ...",
        ktype="solution", project="demo",
    )
    import server as _srv

    recall = _srv.Recall(fast_store)
    result = recall.search(
        query="median", project="demo", limit=10,
        embedding_space="code",  # type: ignore[call-arg]  # arg added in Phase 6b
    )
    items = [i for g in result["results"].values() for i in g]
    for it in items:
        # Phase 6b: every returned row exposes its space in detail mode.
        assert it.get("embedding_space") == "code"


def test_code_chunk_uses_code_space_even_when_only_text_model_configured(
    fast_store, monkeypatch,
):
    """When MEMORY_CODE_EMBED_MODEL is empty, code chunks must fall back to
    the text model BUT still record embedding_space='code'. This is the
    forward-compat contract from §J of the audit. Phase 5b wired the
    classifier + per-space encoder so this is now a hard contract."""
    monkeypatch.setenv("MEMORY_CODE_EMBED_MODEL", "")
    rid, *_ = fast_store.save_knowledge(
        sid="s1",
        content="def add(a: int, b: int) -> int:\n    return a + b\n",
        ktype="solution", project="demo",
    )
    row = fast_store.db.execute(
        "SELECT embedding_space, embed_model FROM embeddings WHERE knowledge_id=?",
        (rid,),
    ).fetchone()
    assert row[0] == "code", f"expected code space, got {row[0]}"
    # The model used falls back to the text model; that's allowed.
    assert row[1], "embedding_model must be recorded even on fallback"


def test_code_save_uses_code_specific_embedding_model_by_default(fast_store):
    """Phase 5b — with default config (MEMORY_CODE_EMBED_MODEL not set,
    so it defaults to jinaai/jina-embeddings-v2-base-code, 768d), a code
    save must produce a 768-dim vector under the code model. Proves the
    per-space encoder really fires."""
    rid, *_ = fast_store.save_knowledge(
        sid="s1",
        content=(
            "def calculate_median(values: list[float]) -> float:\n"
            "    arr = sorted(values)\n"
            "    n = len(arr)\n"
            "    return arr[n//2] if n % 2 else (arr[n//2-1]+arr[n//2])/2\n"
        ),
        ktype="solution", project="demo",
    )
    row = fast_store.db.execute(
        "SELECT embedding_space, embed_model, embed_dim FROM embeddings WHERE knowledge_id=?",
        (rid,),
    ).fetchone()
    assert row[0] == "code"
    assert "code" in (row[1] or "").lower(), (
        f"expected code-aware model, got {row[1]!r}"
    )
    assert row[2] == 768, f"expected 768d (jina-code), got {row[2]}"


# ──────────────────────────────────────────────
# Module-level import guards (for Phase 3/4 — currently xfail)
# ──────────────────────────────────────────────


@pytest.mark.xfail(
    reason="memory_core/ does not exist yet (created in Phase 3). "
    "This guard goes green after Phase 3.",
    strict=False,
)
def test_memory_core_does_not_import_llm_provider():
    """Phase 3+ contract: `src/memory_core/*` must not import `llm_provider`.

    Implementation: walk src/memory_core, parse each .py with `ast`, fail
    if any module has `import llm_provider` or `from llm_provider …`.
    """
    import ast

    core = Path(__file__).parent.parent / "src" / "memory_core"
    if not core.is_dir():
        pytest.skip("memory_core/ not extracted yet (Phase 3)")
    offenders: list[str] = []
    for py in core.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.split(".")[0] == "llm_provider":
                        offenders.append(f"{py.name}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "llm_provider":
                    offenders.append(f"{py.name}:{node.lineno}")
    assert offenders == [], f"memory_core imports llm_provider in: {offenders}"
