"""v11.0 Phase 8 — tests for `bin/memory-perf-gate`.

The gate is a CLI script (no `.py` extension), so we load it via `runpy`
and reach into its namespace. The bench is **never** actually executed in
these tests — we drive `parse_bench_table()` and `evaluate_thresholds()`
with synthetic markdown inputs and stub `run_bench()` for the exit-code
check.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent
GATE_PATH = REPO / "bin" / "memory-perf-gate"


# ──────────────────────────────────────────────
# Module loader
# ──────────────────────────────────────────────


def _load_gate_module():
    """Load bin/memory-perf-gate as a real importlib module so closures
    over its globals (e.g. `run_bench` referenced from inside `main`) see
    monkeypatch overrides correctly. The script has no `.py` suffix, so
    we go through `SourceFileLoader` to bypass extension sniffing."""
    from importlib.machinery import SourceFileLoader

    assert GATE_PATH.is_file(), f"perf-gate script missing: {GATE_PATH}"
    loader = SourceFileLoader("memory_perf_gate", str(GATE_PATH))
    spec = importlib.util.spec_from_loader("memory_perf_gate", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memory_perf_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate_mod():
    return _load_gate_module()


# ──────────────────────────────────────────────
# Synthetic bench output
# ──────────────────────────────────────────────


GREEN_TABLE = """\
| metric              | p50 | p95 | p99 |
|---------------------|-----|-----|-----|
| save_fast           | 12.3 | 18.5 | 22.1 |
| save_fast (cached)  | 1.1  | 2.3  | 3.4  |
| search_fast         | 80.0 | 150.0 | 180.0 |
| cached_search       | 4.0  | 9.0   | 12.0  |
| llm_calls           | 0                |
| network_calls       | 0                |
"""

# search_fast p95 over the 200ms ceiling.
RED_LATENCY = """\
| metric              | p50 | p95 | p99 |
|---------------------|-----|-----|-----|
| save_fast           | 12.3 | 18.5 | 22.1 |
| save_fast (cached)  | 1.1  | 2.3  | 3.4  |
| search_fast         | 180.0 | 280.0 | 320.0 |
| cached_search       | 4.0  | 9.0   | 12.0  |
| llm_calls           | 0                |
| network_calls       | 0                |
"""

# Counter spike — llm_calls != 0.
RED_COUNTER = """\
| metric              | p50 | p95 | p99 |
|---------------------|-----|-----|-----|
| save_fast           | 12.3 | 18.5 | 22.1 |
| save_fast (cached)  | 1.1  | 2.3  | 3.4  |
| search_fast         | 80.0 | 150.0 | 180.0 |
| cached_search       | 4.0  | 9.0   | 12.0  |
| llm_calls           | 3                |
| network_calls       | 0                |
"""


# ──────────────────────────────────────────────
# parse_bench_table
# ──────────────────────────────────────────────


def test_parse_bench_table_extracts_metric_rows(gate_mod):
    parsed = gate_mod.parse_bench_table(GREEN_TABLE)
    assert "save_fast" in parsed
    assert parsed["save_fast"]["p50"] == 12.3
    assert parsed["save_fast"]["p95"] == 18.5
    assert parsed["save_fast"]["p99"] == 22.1
    assert parsed["save_fast (cached)"]["p95"] == 2.3
    assert parsed["search_fast"]["p95"] == 150.0
    assert parsed["cached_search"]["p95"] == 9.0


def test_parse_bench_table_extracts_counter_rows(gate_mod):
    parsed = gate_mod.parse_bench_table(GREEN_TABLE)
    assert parsed["llm_calls"]["value"] == 0
    assert parsed["network_calls"]["value"] == 0


def test_parse_bench_table_skips_separator_and_header(gate_mod):
    parsed = gate_mod.parse_bench_table(GREEN_TABLE)
    assert "metric" not in parsed
    # The separator row `|---|---|...` must not show up as metric "---".
    for k in parsed:
        assert "-" not in set(k.replace("save_fast (cached)", "").replace("save_fast", "")
                              .replace("search_fast", "").replace("cached_search", "")
                              .replace("llm_calls", "").replace("network_calls", ""))


def test_parse_bench_table_handles_empty_input(gate_mod):
    assert gate_mod.parse_bench_table("") == {}
    assert gate_mod.parse_bench_table("no pipes here") == {}


# ──────────────────────────────────────────────
# evaluate_thresholds
# ──────────────────────────────────────────────


def test_evaluate_thresholds_passes_on_green(gate_mod):
    parsed = gate_mod.parse_bench_table(GREEN_TABLE)
    passed, failures, summary = gate_mod.evaluate_thresholds(parsed)
    assert passed is True
    assert failures == []
    # Every threshold metric and zero counter shows up in the summary.
    blob = "\n".join(summary)
    for metric in ("save_fast", "save_fast (cached)", "search_fast",
                   "cached_search", "llm_calls", "network_calls"):
        assert metric in blob, summary


def test_evaluate_thresholds_fails_on_latency_regression(gate_mod):
    parsed = gate_mod.parse_bench_table(RED_LATENCY)
    passed, failures, summary = gate_mod.evaluate_thresholds(parsed)
    assert passed is False
    # Failure mentions the offending metric AND the threshold.
    blob = "\n".join(failures)
    assert "search_fast" in blob
    assert "200" in blob, failures


def test_evaluate_thresholds_fails_on_counter_spike(gate_mod):
    parsed = gate_mod.parse_bench_table(RED_COUNTER)
    passed, failures, summary = gate_mod.evaluate_thresholds(parsed)
    assert passed is False
    blob = "\n".join(failures)
    assert "llm_calls" in blob
    assert "must be 0" in blob


def test_evaluate_thresholds_fails_on_missing_metric(gate_mod):
    # Bench output truncated — drop search_fast row.
    truncated = "\n".join(
        line for line in GREEN_TABLE.splitlines()
        if not line.startswith("| search_fast")
    )
    parsed = gate_mod.parse_bench_table(truncated)
    passed, failures, summary = gate_mod.evaluate_thresholds(parsed)
    assert passed is False
    assert any("search_fast" in f and "missing" in f for f in failures), failures


# ──────────────────────────────────────────────
# main() exit-code behaviour (with run_bench stubbed)
# ──────────────────────────────────────────────


def test_main_exits_zero_on_green(gate_mod, monkeypatch, capsys):
    monkeypatch.setattr(
        gate_mod, "run_bench",
        lambda *_a, **_kw: (0, GREEN_TABLE, ""),
    )
    code = gate_mod.main([])
    captured = capsys.readouterr()
    assert code == 0
    assert "OK:" in captured.out
    assert "FAIL" not in captured.out


def test_main_exits_one_on_latency_regression(gate_mod, monkeypatch, capsys):
    monkeypatch.setattr(
        gate_mod, "run_bench",
        lambda *_a, **_kw: (0, RED_LATENCY, ""),
    )
    code = gate_mod.main([])
    captured = capsys.readouterr()
    assert code == 1
    err_blob = captured.err
    assert "search_fast" in err_blob
    assert "regression" in err_blob.lower()


def test_main_exits_one_on_counter_spike(gate_mod, monkeypatch, capsys):
    monkeypatch.setattr(
        gate_mod, "run_bench",
        lambda *_a, **_kw: (0, RED_COUNTER, ""),
    )
    code = gate_mod.main([])
    captured = capsys.readouterr()
    assert code == 1
    assert "llm_calls" in captured.err


def test_main_propagates_bench_failure(gate_mod, monkeypatch, capsys):
    monkeypatch.setattr(
        gate_mod, "run_bench",
        lambda *_a, **_kw: (2, "partial output", "boom"),
    )
    code = gate_mod.main([])
    captured = capsys.readouterr()
    assert code == 1
    assert "boom" in captured.err
    assert "exited 2" in captured.err


# ──────────────────────────────────────────────
# Threshold values are the documented contract
# ──────────────────────────────────────────────


def test_thresholds_match_phase8_spec(gate_mod):
    """Lock the thresholds to the v11 contract — Phase 8 deliverable (B).

    These values are part of the public CI contract; bumping them must be a
    deliberate code change, not an accidental drift."""
    thr = gate_mod.THRESHOLDS_P95_MS
    assert thr["save_fast"] == 50.0
    assert thr["save_fast (cached)"] == 5.0
    assert thr["search_fast"] == 200.0
    assert thr["cached_search"] == 20.0
    assert set(gate_mod.REQUIRED_ZERO_COUNTERS) == {"llm_calls", "network_calls"}
