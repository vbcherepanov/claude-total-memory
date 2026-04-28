"""W2-I CLI — fit Platt calibration on a validation fixture.

Usage
-----
Generate the synthetic fixture (deterministic seed 42)::

    python scripts/calibrate.py --generate-fixture

Fit a calibrator and write JSON::

    python scripts/calibrate.py \
        --train tests/fixtures/calibration_validation.json \
        --output ~/.claude-memory/calibration.json

Reports
-------
* ECE (10-bin) before fit (raw scores treated as probabilities).
* ECE (10-bin) after fit.
* Suggested per-category thresholds — top of Youden's J on the PR
  curve when there is enough class diversity, with 0.45 fallback.

Why a synthetic fixture
-----------------------
We deliberately do **not** sample LoCoMo's eval split — that would
leak signal into the calibrator and into downstream evaluation. The
synthetic fixture follows the LoCoMo category distribution and uses
beta(5,2) / beta(2,5) for label-conditional scores so the data has
realistic but disentangled structure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make src/ importable when run directly.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from memory_core.calibration import (  # noqa: E402
    apply,
    expected_calibration_error,
    fit_platt,
    save,
)


# LoCoMo category mix used to generate the validation fixture.
# Numbers come from baseline-failure-analysis.md (n column / total).
_CATEGORY_MIX = (
    ("single-hop", 0.14),
    ("multi-hop", 0.16),
    ("temporal", 0.05),
    ("open-domain", 0.42),
    ("adversarial", 0.23),
)


def _generate_fixture(out_path: Path, *, n_examples: int = 200, seed: int = 42) -> None:
    """Generate the deterministic calibration validation fixture.

    For each example:
    * Pick a category by the LoCoMo proportions.
    * Toss a fair coin for the label.
    * Draw the raw score from beta(5,2) when label=1 (skewed high)
      or beta(2,5) when label=0 (skewed low). The two distributions
      overlap heavily so calibration has work to do.

    The header preamble lives in the JSON itself (a top-level
    ``_doc`` key) so anyone opening the file knows where it came
    from and that it must not be derived from LoCoMo eval data.
    """
    rng = np.random.default_rng(seed)

    categories = [c for c, _ in _CATEGORY_MIX]
    weights = np.array([w for _, w in _CATEGORY_MIX], dtype=np.float64)
    weights = weights / weights.sum()

    examples: list[dict[str, object]] = []
    for _ in range(n_examples):
        cat = categories[int(rng.choice(len(categories), p=weights))]
        label = int(rng.integers(0, 2))
        if label == 1:
            score = float(rng.beta(5.0, 2.0))
        else:
            score = float(rng.beta(2.0, 5.0))
        examples.append({"raw_score": score, "label": label, "category": cat})

    payload = {
        "_doc": (
            "Synthetic calibration fixture for W2-I. Do NOT replace with "
            "LoCoMo eval data — that would leak signal. Distribution: "
            "labels ~ Bernoulli(0.5); raw_score ~ Beta(5,2) when label=1 "
            "else Beta(2,5); category sampled by LoCoMo proportions. "
            "Seed=42 for reproducibility."
        ),
        "seed": seed,
        "n_examples": n_examples,
        "examples": examples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def _load_fixture(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read fixture JSON into (scores, labels, categories) arrays."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "examples" not in raw:
        raise ValueError(f"fixture {path} missing top-level 'examples'")
    examples = raw["examples"]
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"fixture {path} has empty 'examples'")
    scores = np.array(
        [float(ex["raw_score"]) for ex in examples], dtype=np.float64
    )
    labels = np.array(
        [int(ex["label"]) for ex in examples], dtype=np.int64
    )
    cats = [str(ex.get("category", "default")) for ex in examples]
    return scores, labels, cats


def _suggest_thresholds(
    probs: np.ndarray, labels: np.ndarray, cats: list[str]
) -> dict[str, float]:
    """Per-category Youden's-J optimum on the calibrated probabilities.

    Youden's J = TPR - FPR; the threshold that maximises it is a
    standard PR-curve operating point. We return a per-category map
    falling back to 0.45 when a category has fewer than 5 samples or
    one of the classes is empty.
    """
    out: dict[str, float] = {}
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(cats):
        by_cat.setdefault(c, []).append(i)
    for cat, idxs in by_cat.items():
        if len(idxs) < 5:
            out[cat] = 0.45
            continue
        p_cat = probs[idxs]
        y_cat = labels[idxs]
        n_pos = int(np.sum(y_cat == 1))
        n_neg = int(np.sum(y_cat == 0))
        if n_pos == 0 or n_neg == 0:
            out[cat] = 0.45
            continue
        # Sweep candidate thresholds on the unique probs, plus midpoints.
        candidates = np.unique(np.concatenate([p_cat, np.linspace(0.05, 0.95, 19)]))
        best_j = -1.0
        best_t = 0.45
        for t in candidates:
            pred = p_cat >= t
            tp = float(np.sum(pred & (y_cat == 1)))
            fn = float(np.sum(~pred & (y_cat == 1)))
            fp = float(np.sum(pred & (y_cat == 0)))
            tn = float(np.sum(~pred & (y_cat == 0)))
            tpr = tp / max(1.0, tp + fn)
            fpr = fp / max(1.0, fp + tn)
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_t = float(t)
        out[cat] = round(best_t, 3)
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit Platt calibration on a validation fixture (W2-I)."
    )
    parser.add_argument(
        "--generate-fixture",
        action="store_true",
        help="Regenerate tests/fixtures/calibration_validation.json (seed=42).",
    )
    parser.add_argument(
        "--train",
        type=Path,
        help="Path to validation fixture JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the fitted calibrator JSON.",
    )
    parser.add_argument(
        "--fixture-out",
        type=Path,
        default=_REPO / "tests" / "fixtures" / "calibration_validation.json",
        help=(
            "Override fixture output path (default: "
            "tests/fixtures/calibration_validation.json)."
        ),
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=200,
        help="Number of synthetic examples (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for fixture generation (default: 42).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.generate_fixture:
        out = args.fixture_out
        _generate_fixture(out, n_examples=args.n_examples, seed=args.seed)
        print(f"[calibrate] wrote fixture → {out} (n={args.n_examples}, seed={args.seed})")
        return 0

    if args.train is None or args.output is None:
        parser.error("--train and --output are required unless --generate-fixture is set")

    scores, labels, cats = _load_fixture(args.train)

    ece_before = expected_calibration_error(scores.astype(np.float64), labels.astype(np.float64))
    cal = fit_platt(scores, labels.astype(np.float64))
    probs = np.array([apply(cal, float(s)) for s in scores], dtype=np.float64)
    ece_after = expected_calibration_error(probs, labels.astype(np.float64))

    output = args.output.expanduser()
    save(cal, output)

    suggested = _suggest_thresholds(probs, labels, cats)

    print(f"[calibrate] fitted Platt: a={cal.a:.4f}, b={cal.b:.4f}")
    print(f"[calibrate] ECE before fit: {ece_before:.4f}")
    print(f"[calibrate] ECE after  fit: {ece_after:.4f}")
    print(f"[calibrate] calibrator → {output}")
    print("[calibrate] suggested per-category thresholds (Youden's J):")
    for cat, t in sorted(suggested.items()):
        print(f"            {cat:<14} {t:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
