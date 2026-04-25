"""Micro-benchmarks for the v9.0 two-level cache.

These are lightweight — each runs a few thousand iterations — and assert
latency SLOs rather than print-only. Run with ``-s`` to see raw numbers:

    pytest tests/test_cache_layer_bench.py -s -q
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _percentiles(samples_ns: list[int]) -> tuple[float, float, float]:
    samples_ns.sort()
    n = len(samples_ns)
    return (
        samples_ns[n // 2] / 1000.0,
        samples_ns[int(n * 0.95)] / 1000.0,
        samples_ns[int(n * 0.99)] / 1000.0,
    )


def test_bench_l1_hit_under_100us(monkeypatch, capsys):
    monkeypatch.setenv("V9_CACHE_L1_ENABLED", "1")
    from cache_layer import L1QueryCache, make_l1_key

    l1 = L1QueryCache(maxsize=1000, ttl_sec=300)
    k = make_l1_key("auth bug", mode="search", k=10, filters={"project": "vito"})
    payload = {"results": [{"id": i, "content": "x" * 120} for i in range(10)]}
    l1.set(k, payload, memory_ids=list(range(10)))

    samples: list[int] = []
    for _ in range(3000):
        t0 = time.perf_counter_ns()
        l1.get(k)
        samples.append(time.perf_counter_ns() - t0)
    p50, p95, p99 = _percentiles(samples)
    with capsys.disabled():
        print(f"\n[bench] L1 hit  p50={p50:.2f}us  p95={p95:.2f}us  p99={p99:.2f}us  n={len(samples)}")
    # Even on a slow CI runner an LRU dict lookup under a lock is sub-ms.
    assert p50 < 100.0, f"L1 hit p50 regressed: {p50:.2f}us"


def test_bench_l2_get_under_3ms(monkeypatch, capsys):
    monkeypatch.setenv("V9_CACHE_L2_ENABLED", "1")
    from cache_layer import L2EmbeddingCache

    with tempfile.TemporaryDirectory() as tmp:
        l2 = L2EmbeddingCache(db_path=os.path.join(tmp, "memory.db"))
        vec = [0.001 * i for i in range(384)]
        for i in range(200):
            l2.set(f"seed-{i}", vec, model="m")

        samples: list[int] = []
        for i in range(1500):
            t0 = time.perf_counter_ns()
            l2.get(f"seed-{i % 200}", expected_dim=384)
            samples.append(time.perf_counter_ns() - t0)
        g50, g95, g99 = _percentiles(samples)
        with capsys.disabled():
            print(f"\n[bench] L2 get  p50={g50:.2f}us  p95={g95:.2f}us  p99={g99:.2f}us  n={len(samples)}")

        writes: list[int] = []
        for i in range(300):
            t0 = time.perf_counter_ns()
            l2.set(f"fresh-{i}", vec, model="m")
            writes.append(time.perf_counter_ns() - t0)
        s50, s95, s99 = _percentiles(writes)
        with capsys.disabled():
            print(f"[bench] L2 set  p50={s50:.2f}us  p95={s95:.2f}us  p99={s99:.2f}us  n={len(writes)}")

        l2.close()
        # Target: L2 get p50 < 3ms. Generous cap for CI variance.
        assert g50 < 3000.0, f"L2 get p50 regressed: {g50:.2f}us"
