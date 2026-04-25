#!/usr/bin/env python3
"""Compare two LongMemEval benchmark result JSONs and gate on R@5 regression.

Supports two schemas:
  A) Legacy/eval schema with `overall.r_at_5_recall_any` and
     `overall.avg_latency_ms_per_query` (e.g. evals/longmemeval-2026-04-17.json).
  B) Bench runner schema with `modes.full.total_r_any` and
     `modes.full.avg_latency_ms` (e.g. benchmarks/results_longmemeval.json).

Exit codes:
  0 — PASS (current R@5 >= baseline R@5 - max_regression)
  1 — FAIL (regression exceeds threshold)
  2 — IO/parse error (file missing or unreadable)

Latency delta is reported but does not affect exit code (warn-only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _extract_metrics(doc: dict) -> tuple[Optional[float], Optional[float]]:
    """Return (r_at_5_recall_any, avg_latency_ms) from either known schema.

    Unknown/missing fields return None. Latency is optional.
    """
    # Schema A: `overall.r_at_5_recall_any`
    overall = doc.get("overall")
    if isinstance(overall, dict) and "r_at_5_recall_any" in overall:
        r = overall.get("r_at_5_recall_any")
        lat = overall.get("avg_latency_ms_per_query")
        return (float(r) if r is not None else None,
                float(lat) if lat is not None else None)

    # Schema B: `modes.full.total_r_any`
    modes = doc.get("modes")
    if isinstance(modes, dict):
        full = modes.get("full")
        if isinstance(full, dict) and "total_r_any" in full:
            r = full.get("total_r_any")
            lat = full.get("avg_latency_ms")
            return (float(r) if r is not None else None,
                    float(lat) if lat is not None else None)

    return (None, None)


def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: {label} file is not valid JSON ({path}): {e}",
              file=sys.stderr)
        sys.exit(2)
    except OSError as e:
        print(f"ERROR: cannot read {label} file {path}: {e}", file=sys.stderr)
        sys.exit(2)


def compare(baseline_path: Path, current_path: Path,
            max_regression: float) -> int:
    baseline = _load_json(baseline_path, "baseline")
    current = _load_json(current_path, "current")

    b_r, b_lat = _extract_metrics(baseline)
    c_r, c_lat = _extract_metrics(current)

    if b_r is None:
        print(f"ERROR: baseline ({baseline_path}) does not contain "
              f"R@5 metric in a known schema.", file=sys.stderr)
        return 2
    if c_r is None:
        print(f"ERROR: current ({current_path}) does not contain "
              f"R@5 metric in a known schema.", file=sys.stderr)
        return 2

    delta = c_r - b_r
    verdict = "PASS" if delta >= -max_regression else "FAIL"

    # Format aligned table for CI log legibility.
    print("LongMemEval R@5 regression check")
    print("-" * 60)
    print(f"  baseline R@5    : {b_r:.4f}  ({baseline_path})")
    print(f"  current  R@5    : {c_r:.4f}  ({current_path})")
    print(f"  delta (pp)      : {delta * 100:+.2f}pp")
    print(f"  threshold (pp)  : -{max_regression * 100:.2f}pp")
    print(f"  verdict         : {verdict}")

    # Latency (warn-only).
    if b_lat is not None and c_lat is not None:
        lat_delta = c_lat - b_lat
        marker = ""
        # 25% slowdown -> visible WARN, no failure
        if b_lat > 0 and lat_delta > 0 and lat_delta / b_lat >= 0.25:
            marker = "  WARN: latency up >=25%"
        print(f"  latency baseline: {b_lat:.2f} ms/q")
        print(f"  latency current : {c_lat:.2f} ms/q   "
              f"(delta {lat_delta:+.2f} ms){marker}")
    print("-" * 60)

    return 0 if verdict == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two LongMemEval result JSONs and gate on "
                    "R@5 regression.")
    p.add_argument("baseline", type=Path,
                   help="Path to baseline result JSON (trusted reference).")
    p.add_argument("current", type=Path,
                   help="Path to current run result JSON.")
    p.add_argument("--max-regression", type=float, default=0.01,
                   help="Maximum tolerated R@5 drop, as absolute fraction "
                        "(default: 0.01 == 1pp).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_regression < 0:
        print("ERROR: --max-regression must be non-negative.", file=sys.stderr)
        return 2
    return compare(args.baseline, args.current, args.max_regression)


if __name__ == "__main__":
    raise SystemExit(main())
