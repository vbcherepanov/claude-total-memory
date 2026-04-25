"""Tests for the v9.0 two-level cache (lane A2).

Covers:
    • L1 hit/miss/TTL/LRU/invalidation semantics.
    • L2 embedding round-trip, dim & model mismatch safety, concurrent writes.
    • Migration 014 applies cleanly on a fresh SQLite DB.
    • Feature-flag gating — cache is a safe no-op when flags are OFF.
    • End-to-end integration sketch: cache hit ratio climbs on repeats.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Reset v9 env between tests to isolate flag state.
@pytest.fixture(autouse=True)
def _reset_v9_env(monkeypatch):
    for k in (
        "V9_CACHE_L1_ENABLED",
        "V9_CACHE_L2_ENABLED",
        "V9_CACHE_L1_SIZE",
        "V9_CACHE_L1_TTL_SEC",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture
def enabled(monkeypatch):
    """Force v9 cache flags ON for a single test."""
    monkeypatch.setenv("V9_CACHE_L1_ENABLED", "1")
    monkeypatch.setenv("V9_CACHE_L2_ENABLED", "1")
    yield


# ──────────────────────────────────────────────────────────────
# L1 behaviour
# ──────────────────────────────────────────────────────────────


def test_l1_hit_returns_cached_value(enabled):
    from cache_layer import L1QueryCache, make_l1_key

    cache = L1QueryCache(maxsize=10, ttl_sec=60)
    k = make_l1_key("auth", mode="search", k=5, filters={"project": "p"})
    cache.set(k, {"hit": True}, memory_ids=[1, 2])
    assert cache.get(k) == {"hit": True}
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_l1_miss_then_fill_then_hit(enabled):
    from cache_layer import L1QueryCache

    cache = L1QueryCache(maxsize=10, ttl_sec=60)
    assert cache.get("key1") is None
    cache.set("key1", {"v": 1})
    assert cache.get("key1") == {"v": 1}
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1


def test_l1_ttl_expire_triggers_miss(enabled, monkeypatch):
    from cache_layer import L1QueryCache
    import cache_layer

    cache = L1QueryCache(maxsize=5, ttl_sec=0.01)
    cache.set("k", "v")
    # Advance time past TTL by patching time.time inside the module.
    original = cache_layer.time.time
    monkeypatch.setattr(cache_layer.time, "time", lambda: original() + 10)
    assert cache.get("k") is None
    # Entry should have been evicted as side-effect.
    assert len(cache) == 0


def test_l1_lru_evicts_oldest_when_over_size(enabled):
    from cache_layer import L1QueryCache

    cache = L1QueryCache(maxsize=3, ttl_sec=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.get("a")  # mark 'a' most recently used
    cache.set("d", 4)  # should evict 'b' (oldest untouched)

    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_l1_invalidate_all_clears_everything(enabled):
    from cache_layer import L1QueryCache

    cache = L1QueryCache(maxsize=10, ttl_sec=60)
    cache.set("x", 1)
    cache.set("y", 2)
    removed = cache.invalidate_all()
    assert removed == 2
    assert cache.get("x") is None
    assert cache.get("y") is None


def test_l1_invalidate_by_id_drops_only_matching(enabled):
    from cache_layer import L1QueryCache

    cache = L1QueryCache(maxsize=10, ttl_sec=60)
    cache.set("k1", "val1", memory_ids=[10, 20])
    cache.set("k2", "val2", memory_ids=[30])
    cache.set("k3", "val3", memory_ids=[10, 40])

    dropped = cache.invalidate_by_id(10)
    assert dropped == 2
    assert cache.get("k1") is None
    assert cache.get("k3") is None
    assert cache.get("k2") == "val2"


# ──────────────────────────────────────────────────────────────
# L2 behaviour
# ──────────────────────────────────────────────────────────────


def test_l2_roundtrip_preserves_vector(enabled, tmp_path):
    from cache_layer import L2EmbeddingCache

    cache = L2EmbeddingCache(db_path=tmp_path / "memory.db")
    vec = [0.1, -0.25, 3.14, 0.0, -0.0001]
    assert cache.set("hello world", vec, model="test-model") is True
    out = cache.get("hello world")
    assert out is not None
    assert len(out) == len(vec)
    for a, b in zip(out, vec):
        assert abs(a - b) < 1e-6
    cache.close()


def test_l2_dim_mismatch_yields_miss(enabled, tmp_path):
    from cache_layer import L2EmbeddingCache

    cache = L2EmbeddingCache(db_path=tmp_path / "memory.db")
    cache.set("t", [1.0, 2.0, 3.0, 4.0], model="m")
    # Caller demands dim=8 — returned shape is 4, must be treated as miss.
    assert cache.get("t", expected_dim=8) is None
    # Correct dim works.
    assert cache.get("t", expected_dim=4) == pytest.approx([1.0, 2.0, 3.0, 4.0])
    # Wrong model name also yields miss.
    assert cache.get("t", expected_model="other") is None
    cache.close()


def test_l2_concurrent_writes_do_not_crash(enabled, tmp_path):
    from cache_layer import L2EmbeddingCache

    cache = L2EmbeddingCache(db_path=tmp_path / "memory.db")
    errors: list[BaseException] = []

    def worker(prefix: str):
        try:
            for i in range(50):
                cache.set(f"{prefix}-{i}", [float(i)] * 8, model="m")
        except BaseException as e:  # pragma: no cover - diagnostic only
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert errors == []
    # Either thread's write may have landed; we just need 100 rows readable.
    assert cache.size() == 100
    cache.close()


def test_migration_014_applies_on_fresh_db(tmp_path):
    """The SQL migration file should create embedding_cache idempotently."""
    mig = Path(__file__).resolve().parent.parent / "migrations" / "014_embedding_cache.sql"
    assert mig.exists(), "migration file must exist"

    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    # Seed the migrations tracker so INSERT OR IGNORE doesn't fail.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            version TEXT PRIMARY KEY,
            description TEXT NOT NULL
        );
        """
    )
    # Apply twice — must be idempotent.
    sql = mig.read_text()
    conn.executescript(sql)
    conn.executescript(sql)

    # Table exists with expected columns.
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
    ).fetchone()
    assert row is not None
    cols = {r[1] for r in conn.execute("PRAGMA table_info(embedding_cache)").fetchall()}
    assert cols == {"key", "embedding", "created_at", "model", "dim"}

    # Migration tracker populated.
    applied = conn.execute(
        "SELECT description FROM migrations WHERE version=?", ("014",)
    ).fetchone()
    assert applied is not None
    conn.close()


# ──────────────────────────────────────────────────────────────
# Integration helpers
# ──────────────────────────────────────────────────────────────


def test_cache_hit_ratio_climbs_on_repeats(enabled):
    from cache_layer import L1QueryCache, make_l1_key

    cache = L1QueryCache(maxsize=100, ttl_sec=60)
    # Warm the cache.
    for i in range(10):
        cache.set(make_l1_key(f"q{i}", mode="m", k=5, filters={}), {"i": i})

    # Repeated hits on the same 10 queries drive ratio toward 1.0.
    for _ in range(3):
        for i in range(10):
            out = cache.get(make_l1_key(f"q{i}", mode="m", k=5, filters={}))
            assert out == {"i": i}

    stats = cache.stats()
    assert stats["hit_ratio"] >= 0.95


def test_invalidation_on_memory_save_drops_related_l1_entries(enabled):
    """Simulates what server.py does on memory_save: invalidate_all or by_id."""
    from cache_layer import TwoLevelCache

    cache = TwoLevelCache(db_path=":memory:")
    cache.recall_set("auth bug", {"res": "a"}, mode="search", k=5,
                     filters={"project": "p"}, memory_ids=[42])
    cache.recall_set("billing", {"res": "b"}, mode="search", k=5,
                     filters={"project": "p"}, memory_ids=[99])

    # A save touches record 42 → invalidate_by_id drops only its entry.
    dropped = cache.invalidate_by_id(42)
    assert dropped == 1
    assert cache.recall_get("auth bug", mode="search", k=5, filters={"project": "p"}) is None
    assert cache.recall_get("billing", mode="search", k=5, filters={"project": "p"}) is not None

    # A full invalidate (generic save path) drops the rest.
    cache.invalidate_all()
    assert cache.recall_get("billing", mode="search", k=5, filters={"project": "p"}) is None


# ──────────────────────────────────────────────────────────────
# Flag gating — cache must be a no-op when disabled
# ──────────────────────────────────────────────────────────────


def test_flags_off_makes_cache_a_noop(monkeypatch, tmp_path):
    """Default env: both flags OFF → get always None, set does nothing."""
    for k in ("V9_CACHE_L1_ENABLED", "V9_CACHE_L2_ENABLED"):
        monkeypatch.delenv(k, raising=False)

    from cache_layer import L1QueryCache, L2EmbeddingCache

    l1 = L1QueryCache(maxsize=10, ttl_sec=60)
    l1.set("k", "v")
    assert l1.get("k") is None  # no-op
    assert len(l1) == 0

    l2 = L2EmbeddingCache(db_path=tmp_path / "m.db")
    assert l2.set("t", [1.0, 2.0], model="m") is False
    assert l2.get("t") is None
    l2.close()


def test_flag_on_without_env_uses_safe_defaults(monkeypatch, tmp_path):
    """When flag is explicitly ON but SIZE/TTL envs absent, defaults kick in."""
    monkeypatch.setenv("V9_CACHE_L1_ENABLED", "1")
    monkeypatch.delenv("V9_CACHE_L1_SIZE", raising=False)
    monkeypatch.delenv("V9_CACHE_L1_TTL_SEC", raising=False)

    from cache_layer import L1QueryCache

    cache = L1QueryCache()
    stats = cache.stats()
    assert stats["maxsize"] == 1000
    assert stats["ttl_sec"] == 300.0


def test_facade_stats_include_both_tiers(enabled, tmp_path):
    from cache_layer import TwoLevelCache

    cache = TwoLevelCache(db_path=tmp_path / "m.db")
    cache.recall_set("q", {"r": 1}, mode="s", k=5, filters={})
    cache.embed_set("text", [1.0, 2.0, 3.0], model="m")
    s = cache.stats()
    assert "l1" in s and "l2_size" in s
    assert s["l2_size"] >= 1
    assert s["l1"]["size"] >= 1
    cache.close()
