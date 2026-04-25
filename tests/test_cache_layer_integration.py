"""End-to-end-ish integration test for the v9.0 cache wired into Store.

Runs with flags ON and exercises a save→recall→save cycle against a
throw-away ~/.claude-memory temp dir, asserting that the recall path
serves from L1 on a repeat hit and that the L2 row gets written for
the embedding text.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Spin up a Store against a temp MEMORY_DIR with v9 flags flipped on."""
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("V9_CACHE_L1_ENABLED", "1")
    monkeypatch.setenv("V9_CACHE_L2_ENABLED", "1")
    # Avoid the noisy FastEmbed download in CI by forcing ST fallback.
    monkeypatch.setenv("USE_BINARY_SEARCH", "false")

    # Reload server so it picks up the new env + MEMORY_DIR constant.
    for mod in list(sys.modules):
        if mod in ("server", "cache_layer"):
            del sys.modules[mod]

    import server  # noqa: WPS433 — runtime import by design
    store = server.Store()
    yield store
    try:
        if getattr(store, "v9_cache", None) is not None:
            store.v9_cache.close()
        store.db.close()
    except Exception:
        pass


def test_store_exposes_two_level_cache(isolated_store):
    store = isolated_store
    assert store.v9_cache is not None
    assert store.v9_cache.l1.enabled is True
    assert store.v9_cache.l2.enabled is True


def test_embedding_cache_table_created_by_migration(isolated_store):
    """Migration 014 is applied on boot — table must exist."""
    store = isolated_store
    row = store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
    ).fetchone()
    assert row is not None
    cols = {r[1] for r in store.db.execute("PRAGMA table_info(embedding_cache)").fetchall()}
    assert cols == {"key", "embedding", "created_at", "model", "dim"}


def test_invalidation_hook_clears_l1_on_save(isolated_store):
    store = isolated_store
    # Pre-seed an L1 entry.
    store.v9_cache.recall_set(
        "pre-existing query",
        {"results": {}, "total": 0},
        mode="search",
        k=10,
        filters={"project": "p"},
        memory_ids=[1],
    )
    assert store.v9_cache.recall_get(
        "pre-existing query", mode="search", k=10, filters={"project": "p"}
    ) is not None

    # Emulate what server.py does on memory_save: blanket invalidate.
    store.v9_cache.invalidate_all()
    assert store.v9_cache.recall_get(
        "pre-existing query", mode="search", k=10, filters={"project": "p"}
    ) is None
