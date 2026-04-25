"""Unit tests for scripts/compare_eval.py — the nightly LongMemEval regression gate.

Covers both supported JSON schemas (`overall.r_at_5_recall_any` and
`modes.full.total_r_any`), threshold behaviour, graceful IO failures, and the
latency warn-only branch.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_eval.py"


def _run_script(*argv: str) -> int:
    """Execute compare_eval.py as a CLI, return its exit code."""
    old_argv = sys.argv
    sys.argv = ["compare_eval.py", *argv]
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # Non-int exit codes should be treated as failure.
        return 1
    finally:
        sys.argv = old_argv


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _schema_bench(r_any: float, latency_ms: float = 38.0) -> dict:
    """Bench runner schema (produced by longmemeval_bench.py)."""
    return {
        "k": 5,
        "total_questions": 470,
        "modes": {
            "full": {
                "total_r_any": r_any,
                "total_ndcg": 0.82,
                "total_r_all": 0.84,
                "avg_latency_ms": latency_ms,
                "per_type": {},
            }
        },
    }


def _schema_eval(r_any: float, latency_ms: float = 38.0) -> dict:
    """Legacy `evals/*.json` schema with `overall.r_at_5_recall_any`."""
    return {
        "benchmark": "LongMemEval (public)",
        "k": 5,
        "overall": {
            "r_at_5_recall_any": r_any,
            "r_at_5_recall_all": 0.84,
            "ndcg_at_5": 0.82,
            "avg_latency_ms_per_query": latency_ms,
        },
    }


def test_equal_baselines_pass_exit_zero(tmp_path: Path) -> None:
    base = _write_json(tmp_path / "baseline.json", _schema_bench(0.9617))
    curr = _write_json(tmp_path / "current.json", _schema_bench(0.9617))
    assert _run_script(str(base), str(curr), "--max-regression", "0.01") == 0


def test_current_5pp_above_baseline_pass(tmp_path: Path, capsys) -> None:
    base = _write_json(tmp_path / "baseline.json", _schema_bench(0.90))
    curr = _write_json(tmp_path / "current.json", _schema_bench(0.95))
    code = _run_script(str(base), str(curr), "--max-regression", "0.01")
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "+5.00pp" in out


def test_regression_within_threshold_pass(tmp_path: Path, capsys) -> None:
    # 0.5pp drop with 1pp threshold — PASS.
    base = _write_json(tmp_path / "baseline.json", _schema_bench(0.960))
    curr = _write_json(tmp_path / "current.json", _schema_bench(0.955))
    code = _run_script(str(base), str(curr), "--max-regression", "0.01")
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_regression_exceeds_threshold_fail(tmp_path: Path, capsys) -> None:
    # 2pp drop with 1pp threshold — FAIL exit 1.
    base = _write_json(tmp_path / "baseline.json", _schema_bench(0.960))
    curr = _write_json(tmp_path / "current.json", _schema_bench(0.940))
    code = _run_script(str(base), str(curr), "--max-regression", "0.01")
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_missing_current_file_exit_two(tmp_path: Path, capsys) -> None:
    base = _write_json(tmp_path / "baseline.json", _schema_bench(0.96))
    missing = tmp_path / "does_not_exist.json"
    code = _run_script(str(base), str(missing), "--max-regression", "0.01")
    err = capsys.readouterr().err
    assert code == 2
    assert "not found" in err
    assert str(missing) in err


def test_invalid_json_exit_two(tmp_path: Path, capsys) -> None:
    base = _write_json(tmp_path / "baseline.json", _schema_bench(0.96))
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    code = _run_script(str(base), str(bad), "--max-regression", "0.01")
    err = capsys.readouterr().err
    assert code == 2
    assert "not valid JSON" in err


def test_latency_degradation_warn_only(tmp_path: Path, capsys) -> None:
    # Same R@5 but latency doubled — still PASS (warn-only).
    base = _write_json(tmp_path / "baseline.json", _schema_bench(0.96, 40.0))
    curr = _write_json(tmp_path / "current.json", _schema_bench(0.96, 80.0))
    code = _run_script(str(base), str(curr), "--max-regression", "0.01")
    assert code == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "WARN" in out
    assert "latency up" in out


def test_cross_schema_baseline_eval_current_bench(tmp_path: Path, capsys) -> None:
    # Baseline is the legacy eval schema, current is the bench runner schema.
    base = _write_json(tmp_path / "baseline.json", _schema_eval(0.9617, 38.77))
    curr = _write_json(tmp_path / "current.json", _schema_bench(0.9617, 38.10))
    code = _run_script(str(base), str(curr), "--max-regression", "0.01")
    assert code == 0
    out = capsys.readouterr().out
    assert "0.9617" in out


def test_baseline_missing_r_at_5_exit_two(tmp_path: Path, capsys) -> None:
    base = _write_json(tmp_path / "baseline.json", {"overall": {}})
    curr = _write_json(tmp_path / "current.json", _schema_bench(0.96))
    code = _run_script(str(base), str(curr), "--max-regression", "0.01")
    err = capsys.readouterr().err
    assert code == 2
    assert "R@5" in err
