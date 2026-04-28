#!/usr/bin/env python3
"""v9.0 D6 — mine (question, gold_answer) few-shot pairs from LoCoMo train.

Strategy (v2):
  * Read locomo10.json.
  * Restrict to --train-conv-ids (default: first 70%, same convention as D5).
  * Group QAs by category (1..5). For each cat, pick the most "representative"
    pairs — short answers, distinct head nouns, diverse questions.
  * Output benchmarks/data/locomo_few_shot_v2.json that the bench loads when
    --few-shot-pairs is set.

Strategy (v3, --version 3):
  * Same dataset / leakage-free split.
  * For each category sample 30 pairs uniformly (seed=42), then prune to 15
    by maximising distinct question stems (regex on first interrogative)
    AND distinct answer surface forms (named entities / dates / numbers /
    lists / free-text), deduping canonicalised answers.
  * Sort final 15 lexicographically by question for deterministic output.
  * v3 covers cat 5 by falling back to the dataset's `adversarial_answer`
    field when `answer` is empty (LoCoMo schema quirk).

Output schema (identical for v2 and v3):
  {
    "1": [{"q": "...", "a": "..."}, ...],
    "2": [...],
    ...
    "_meta": {
      "train_conv_ids": [0,1,...],
      "n_per_category": 15,
      "source": "benchmarks/data/locomo/data/locomo10.json",
      "created_at": "...",
      "version": 3,
      "seed": 42
    }
  }

Usage:
    python scripts/mine_locomo_fewshot.py \
        --train-conv-ids 0,1,2,3,4,5,6 \
        --n-per-category 20 \
        --output benchmarks/data/locomo_few_shot_v2.json

    python scripts/mine_locomo_fewshot.py \
        --version 3 \
        --output benchmarks/data/locomo_few_shot_v3.json \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "data" / "locomo_few_shot_v2.json"
DEFAULT_OUTPUT_V3 = ROOT / "benchmarks" / "data" / "locomo_few_shot_v3.json"

V3_N_PER_CATEGORY = 15
V3_SAMPLE_POOL = 30
V3_SEED_DEFAULT = 42


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


# ------------------------------------------------------------------ v3 helpers


_QUESTION_STEM_RE = re.compile(
    r"^\s*(who|whom|whose|what|which|where|when|why|how\s+\w+|how|did|do|does|is|are|was|were|will|would|could|should|has|have|had|can|may|might|in|on)\b",
    re.IGNORECASE,
)


def _question_stem(question: str) -> str:
    """Canonical 'question stem' for diversity bucketing.

    Returns the first interrogative phrase in lowercase, e.g.
        "Where did Alice go?"        -> "where"
        "How many dogs does X have?" -> "how many"
        "Did Alice meet Bob?"        -> "did"
    Falls back to the first word lowercased when no canonical interrogative
    matches (covers e.g. "Who's...", contractions).
    """
    m = _QUESTION_STEM_RE.match(question or "")
    if not m:
        first = (question or "").strip().split()
        return first[0].lower().rstrip("?,.:;") if first else ""
    return re.sub(r"\s+", " ", m.group(1).strip().lower())


_NUMBER_RE = re.compile(r"^\s*-?\d+([.,]\d+)?\s*$")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DATE_HINT_RE = re.compile(
    r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sept(ember)?|oct(ober)?|nov(ember)?|dec(ember)?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
}
_YESNO_TOKENS = {"yes", "no", "likely yes", "likely no", "probably yes", "probably no"}


def _answer_surface_form(answer: str) -> str:
    """Bucket the answer into a coarse surface-form class for diversity."""
    raw = (answer or "").strip()
    low = raw.lower().rstrip(".!?,;:")
    if not raw:
        return "empty"
    if low in _YESNO_TOKENS:
        return "yesno"
    if _YEAR_RE.search(raw):
        return "year"
    if _NUMBER_RE.match(raw) or low in _NUMBER_WORDS:
        return "number"
    if _DATE_HINT_RE.search(raw):
        return "date"
    if "," in raw or " and " in low or ";" in raw:
        return "list"
    n_words = len(raw.split())
    if n_words == 1:
        return "entity"
    if n_words <= 4:
        return "short_phrase"
    return "free_text"


def _canonical_answer(answer: str) -> str:
    """Lowercase + strip whitespace + drop trailing punctuation for dedupe."""
    return re.sub(r"[\s\W]+", " ", (answer or "").lower()).strip()


def _collect_pairs_v3(raw: list, train_conv_ids: set[int]) -> dict[int, list[dict]]:
    """Collect every legal (q, a) pair grouped by category for v3 mining.

    For category 5 we accept `adversarial_answer` when `answer` is empty
    (LoCoMo dataset quirk — adversarial questions store the gold under a
    different key).
    """
    by_cat: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for sample_idx, sample in enumerate(raw):
        if sample_idx not in train_conv_ids:
            continue
        for qa_idx, qa in enumerate(sample.get("qa", [])):
            cat = qa.get("category")
            if cat not in by_cat:
                continue
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            if not a and cat == 5:
                a = str(qa.get("adversarial_answer", "")).strip()
            if not q or not a:
                continue
            by_cat[cat].append({
                "q": q,
                "a": a,
                "sample_idx": sample_idx,
                "qa_idx": qa_idx,
            })
    return by_cat


def _select_diverse_v3(pool: list[dict], k: int) -> list[dict]:
    """Pick k pairs from the pool maximising stem and surface-form diversity.

    Pool is iterated in stable (input) order. For each pair we compute its
    (stem, surface_form, canonical_answer) triple. Greedy two-pass:
      1. First pass — accept a pair only if it brings either a previously
         unseen stem OR a previously unseen surface form (and the canonical
         answer is unique within the picked set).
      2. Second pass (top-up) — if we still have <k, accept anything with a
         unique canonical answer.
      3. Final top-up — accept anything not already picked, even duplicate
         canonical answers, until we hit k or the pool is exhausted.
    """
    picked: list[dict] = []
    seen_stems: set[str] = set()
    seen_surfaces: set[str] = set()
    seen_canon: set[str] = set()

    annotated = []
    for p in pool:
        annotated.append({
            **p,
            "_stem": _question_stem(p["q"]),
            "_surface": _answer_surface_form(p["a"]),
            "_canon": _canonical_answer(p["a"]),
        })

    # Pass 1 — strictly diverse on (stem OR surface) and unique canon.
    for p in annotated:
        if len(picked) >= k:
            break
        if p["_canon"] in seen_canon:
            continue
        is_new_stem = p["_stem"] not in seen_stems
        is_new_surface = p["_surface"] not in seen_surfaces
        if not (is_new_stem or is_new_surface):
            continue
        picked.append(p)
        seen_stems.add(p["_stem"])
        seen_surfaces.add(p["_surface"])
        seen_canon.add(p["_canon"])

    # Pass 2 — accept anything with a still-unique canon answer.
    if len(picked) < k:
        for p in annotated:
            if len(picked) >= k:
                break
            if p in picked:
                continue
            if p["_canon"] in seen_canon:
                continue
            picked.append(p)
            seen_stems.add(p["_stem"])
            seen_surfaces.add(p["_surface"])
            seen_canon.add(p["_canon"])

    # Pass 3 — last-resort top-up (allows canon duplicates for tiny pools).
    if len(picked) < k:
        for p in annotated:
            if len(picked) >= k:
                break
            if p in picked:
                continue
            picked.append(p)

    return picked[:k]


def mine_v3(
    dataset_path: Path,
    train_conv_ids: list[int],
    n_per_category: int = V3_N_PER_CATEGORY,
    sample_pool: int = V3_SAMPLE_POOL,
    seed: int = V3_SEED_DEFAULT,
) -> dict:
    """v3 mining — uniform sample → diversity prune → lex sort.

    Deterministic for a fixed (dataset_path content, train_conv_ids, seed).
    """
    raw = json.loads(dataset_path.read_text())
    train_set = set(train_conv_ids)
    by_cat = _collect_pairs_v3(raw, train_set)

    bundle: dict = {}
    for cat in (1, 2, 3, 4, 5):
        items = by_cat[cat]
        # Stable sort by (sample_idx, qa_idx) so seeded shuffle is reproducible.
        items.sort(key=lambda p: (p["sample_idx"], p["qa_idx"], p["q"]))
        # Per-category RNG so adding a category doesn't reshuffle others.
        rnd = random.Random(f"{seed}-{cat}")
        if len(items) <= sample_pool:
            sampled = list(items)
            rnd.shuffle(sampled)
        else:
            sampled = rnd.sample(items, sample_pool)
        picked = _select_diverse_v3(sampled, n_per_category)
        # Lex sort by question for deterministic on-disk output.
        picked.sort(key=lambda p: p["q"])
        bundle[str(cat)] = [{"q": p["q"], "a": p["a"]} for p in picked]

    bundle["_meta"] = {
        "train_conv_ids": sorted(train_set),
        "n_per_category": n_per_category,
        "source": (
            str(dataset_path.relative_to(ROOT))
            if dataset_path.is_relative_to(ROOT)
            else str(dataset_path)
        ),
        "created_at": "2026-04-28T00:00:00Z",
        "version": 3,
        "seed": seed,
        "sample_pool": sample_pool,
    }
    return bundle


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Defaults depend on --version (v2 or v3).")
    p.add_argument("--train-conv-ids", type=str, default=None,
                   help="Comma-separated conv ids (default: first 70%).")
    p.add_argument("--n-per-category", type=int, default=None,
                   help="Pairs per category (v2 default 20, v3 default 15).")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (v2 default 1337, v3 default 42).")
    p.add_argument("--version", type=int, choices=(2, 3), default=2,
                   help="Mining strategy version (default 2 keeps legacy CLI).")
    p.add_argument("--sample-pool", type=int, default=V3_SAMPLE_POOL,
                   help="v3 only: candidate pool size before diversity prune.")
    args = p.parse_args()

    raw = json.loads(args.dataset.read_text())
    train_ids = _parse_conv_ids(args.train_conv_ids, len(raw))
    print(f"[fewshot] v{args.version} mining from {len(train_ids)} train conversations: {train_ids}")

    if args.version == 3:
        out_path = args.output or DEFAULT_OUTPUT_V3
        n_per = args.n_per_category if args.n_per_category is not None else V3_N_PER_CATEGORY
        seed = args.seed if args.seed is not None else V3_SEED_DEFAULT
        bundle = mine_v3(args.dataset, train_ids, n_per, args.sample_pool, seed)
    else:
        out_path = args.output or DEFAULT_OUTPUT
        n_per = args.n_per_category if args.n_per_category is not None else 20
        seed = args.seed if args.seed is not None else 1337
        bundle = mine(args.dataset, train_ids, n_per, seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    counts = {k: len(v) for k, v in bundle.items() if k != "_meta"}
    print(f"[fewshot] wrote {sum(counts.values())} pairs → {out_path}")
    print(f"[fewshot] per-category: {counts}")
    held_out = sorted(set(range(len(raw))) - set(train_ids))
    print(f"[fewshot] held-out conv ids (legal eval): {held_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
