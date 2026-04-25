#!/usr/bin/env python3
"""v9.0 D6 — mine (question, gold_answer) few-shot pairs from LoCoMo train.

Strategy:
  * Read locomo10.json.
  * Restrict to --train-conv-ids (default: first 70%, same convention as D5).
  * Group QAs by category (1..5). For each cat, pick the most "representative"
    pairs — short answers, distinct head nouns, diverse questions.
  * Output benchmarks/data/locomo_few_shot_v2.json that the bench loads when
    --few-shot-pairs is set.

Output schema:
  {
    "1": [{"q": "...", "a": "..."}, ...],
    "2": [...],
    ...
    "_meta": {
      "train_conv_ids": [0,1,...],
      "n_per_category": 20,
      "source": "benchmarks/data/locomo/data/locomo10.json",
      "created_at": "..."
    }
  }

Usage:
    python scripts/mine_locomo_fewshot.py \
        --train-conv-ids 0,1,2,3,4,5,6 \
        --n-per-category 20 \
        --output benchmarks/data/locomo_few_shot_v2.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "data" / "locomo_few_shot_v2.json"


def _parse_conv_ids(raw: str | None, n_total: int) -> list[int]:
    if not raw:
        cutoff = max(1, int(n_total * 0.7))
        return list(range(cutoff))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _short_answer_score(answer: str) -> float:
    """Lower score = better few-shot candidate.

    LoCoMo gold answers are typically 1-6 word noun phrases. Penalize long,
    multi-clause answers (those tend to be open-domain explanations that
    don't help guide the LLM toward the canonical surface form).
    """
    if not answer:
        return 1e6
    n_words = len(answer.split())
    return n_words + 0.1 * len(answer)


def _dedupe_by_question_prefix(pairs: list[dict], k: int) -> list[dict]:
    """Try to maximise question diversity by avoiding near-duplicates."""
    out: list[dict] = []
    seen_prefixes: set[str] = set()
    for p in pairs:
        # First 3 words is a cheap proxy for question type ("Where did Alice..."
        # / "When did Alice..." / "What does Alice...").
        prefix = " ".join(p["q"].lower().split()[:3])
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        out.append(p)
        if len(out) >= k:
            break
    # If diversity filter was too aggressive, top up with whatever's left.
    if len(out) < k:
        for p in pairs:
            if p in out:
                continue
            out.append(p)
            if len(out) >= k:
                break
    return out[:k]


def mine(
    dataset_path: Path,
    train_conv_ids: list[int],
    n_per_category: int,
    seed: int = 1337,
) -> dict:
    raw = json.loads(dataset_path.read_text())
    rnd = random.Random(seed)

    by_cat: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for sample_idx, sample in enumerate(raw):
        if sample_idx not in train_conv_ids:
            continue
        for qa in sample.get("qa", []):
            cat = qa.get("category")
            if cat not in by_cat:
                continue
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            if not q or not a:
                continue
            if cat == 5:
                # Adversarial: gold often "Not mentioned"-style; we accept those.
                pass
            by_cat[cat].append({"q": q, "a": a, "sample_idx": sample_idx})

    out: dict = {}
    for cat, items in by_cat.items():
        items.sort(key=lambda p: _short_answer_score(p["a"]))
        rnd.shuffle(items[: n_per_category * 4])  # mild shuffle within best slice
        picked = _dedupe_by_question_prefix(items, n_per_category)
        out[str(cat)] = [{"q": p["q"], "a": p["a"]} for p in picked]

    out["_meta"] = {
        "train_conv_ids": sorted(set(train_conv_ids)),
        "n_per_category": n_per_category,
        "source": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--train-conv-ids", type=str, default=None,
                   help="Comma-separated conv ids (default: first 70%).")
    p.add_argument("--n-per-category", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    raw = json.loads(args.dataset.read_text())
    train_ids = _parse_conv_ids(args.train_conv_ids, len(raw))
    print(f"[fewshot] mining from {len(train_ids)} train conversations: {train_ids}")

    bundle = mine(args.dataset, train_ids, args.n_per_category, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    counts = {k: len(v) for k, v in bundle.items() if k != "_meta"}
    print(f"[fewshot] wrote {sum(counts.values())} pairs → {args.output}")
    print(f"[fewshot] per-category: {counts}")
    held_out = sorted(set(range(len(raw))) - set(train_ids))
    print(f"[fewshot] held-out conv ids (legal eval): {held_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
