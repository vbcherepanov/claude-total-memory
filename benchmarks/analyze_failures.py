#!/usr/bin/env python3
"""LoCoMo failure analyzer.

Reads a benchmark result JSON (produced by locomo_bench_llm.py) and classifies
each incorrect record into one of:
  - retrieval_miss   : gold evidence is not in retrieved_dia_ids (R@K = 0)
  - over_cautious    : evidence found (R@K >= 1) but pred is an "I don't know"-style refusal
  - hallucination    : evidence found, pred is non-empty, but mismatches gold (LLM ignored or warped evidence)
  - judge_disagree   : pred is semantically close to gold (high f1/rouge_l) yet judge=False
  - format_drift     : pred differs from gold only by formatting (case, punct, "Not"/"none of")
  - unknown          : doesn't fit any heuristic — flagged for manual review

Output:
  - per-category histogram (table)
  - CSV with all wrong rows + assigned bucket
  - top-K examples per bucket per category to stderr

Usage:
    python benchmarks/analyze_failures.py benchmarks/results/v9-paper-methodology-gpt4o.json
    python benchmarks/analyze_failures.py RESULT.json --csv out.csv --top 5
    python benchmarks/analyze_failures.py A.json --diff B.json    # show regressions A→B
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REFUSAL_PATTERNS = re.compile(
    r"\b("
    r"not (mentioned|stated|provided|available|specified|discussed|in the (conversation|context|memory))"
    r"|no (information|mention|record|details|evidence)"
    r"|cannot (be )?(determined|answered|inferred)"
    r"|unable to (determine|answer|find)"
    r"|i (don't|do not) (know|have)"
    r"|insufficient (information|context|evidence)"
    r"|unknown|unclear"
    r"|не (упомина|сказан|указан|обсужда)"
    r"|нет (информации|данных|упоминан)"
    r"|не знаю|неизвестно"
    r")\b",
    re.IGNORECASE,
)

# adversarial gold answers commonly contain phrases like "no information",
# "not enough", etc. We use this to distinguish a CORRECT refusal (matches
# adversarial gold) from an OVER-CAUTIOUS refusal (gold has a real answer).
ADVERSARIAL_GOLD = re.compile(
    r"\b(no information|not (mentioned|enough|provided)|cannot|unknown|none)\b",
    re.IGNORECASE,
)


CATEGORY_LABELS = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}


def is_refusal(text: str) -> bool:
    if not text:
        return True
    return bool(REFUSAL_PATTERNS.search(text))


def is_adversarial_gold(text: str) -> bool:
    if not text:
        return True
    return bool(ADVERSARIAL_GOLD.search(text))


def has_retrieval(rec: dict) -> bool:
    """True if any gold evidence dia-id was retrieved within top-10."""
    # the bench stores r@1/r@5/r@10 as 0/1 floats indicating recall_any.
    for k in ("r@10", "r@5", "r@1"):
        v = rec.get(k)
        if v is not None:
            try:
                return float(v) > 0
            except (TypeError, ValueError):
                continue
    return False


def normalize(text: str) -> str:
    """Lowercase, strip non-alphanumeric, collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def is_format_drift(pred: str, gold: str) -> bool:
    """Same content, different surface."""
    np_, ng = normalize(pred), normalize(gold)
    if not np_ or not ng:
        return False
    if np_ == ng:
        return True
    # one contains the other (e.g. "Berlin" vs "She lives in Berlin")
    if ng in np_ or np_ in ng:
        # require at least 80% overlap by character length
        short, long = sorted([np_, ng], key=len)
        if len(short) / max(len(long), 1) >= 0.5:
            return True
    return False


def classify(rec: dict) -> str:
    pred = rec.get("pred") or ""
    gold = rec.get("gold") or ""
    f1 = float(rec.get("f1") or 0.0)
    rouge = float(rec.get("rouge_l") or 0.0)
    retrieval_ok = has_retrieval(rec)
    pred_refuses = is_refusal(pred)
    gold_says_no = is_adversarial_gold(gold)

    # 1. retrieval miss — gold evidence not retrieved at all
    if not retrieval_ok:
        return "retrieval_miss"

    # 2. over-cautious — found relevant docs but refused (and gold has real content)
    if pred_refuses and not gold_says_no:
        return "over_cautious"

    # 3. format drift — same answer, different surface
    if is_format_drift(pred, gold):
        return "format_drift"

    # 4. judge_disagree — high token-overlap suggests pred is correct in spirit
    if f1 >= 0.6 or rouge >= 0.6:
        return "judge_disagree"

    # 5. hallucination — evidence was retrieved, pred is confident, mismatches gold
    if retrieval_ok and not pred_refuses:
        return "hallucination"

    return "unknown"


def analyze(records: list[dict]) -> dict:
    by_cat_total: Counter = Counter()
    by_cat_correct: Counter = Counter()
    by_cat_bucket: dict[int, Counter] = defaultdict(Counter)
    examples: dict[tuple[int, str], list[dict]] = defaultdict(list)
    classified: list[dict] = []

    for rec in records:
        cat = int(rec.get("category") or 0)
        by_cat_total[cat] += 1
        if rec.get("correct"):
            by_cat_correct[cat] += 1
            continue
        bucket = classify(rec)
        by_cat_bucket[cat][bucket] += 1
        examples[(cat, bucket)].append(rec)
        out = {
            "category": cat,
            "category_label": CATEGORY_LABELS.get(cat, f"cat_{cat}"),
            "bucket": bucket,
            "question": rec.get("question", ""),
            "gold": rec.get("gold", ""),
            "pred": rec.get("pred", ""),
            "f1": rec.get("f1", 0),
            "rouge_l": rec.get("rouge_l", 0),
            "r@1": rec.get("r@1", 0),
            "r@5": rec.get("r@5", 0),
            "r@10": rec.get("r@10", 0),
        }
        classified.append(out)

    return {
        "by_cat_total": by_cat_total,
        "by_cat_correct": by_cat_correct,
        "by_cat_bucket": by_cat_bucket,
        "examples": examples,
        "classified": classified,
    }


def print_table(result: dict, file=sys.stdout) -> None:
    total = result["by_cat_total"]
    correct = result["by_cat_correct"]
    buckets = result["by_cat_bucket"]

    bucket_names = [
        "retrieval_miss",
        "over_cautious",
        "hallucination",
        "judge_disagree",
        "format_drift",
        "unknown",
    ]
    header = ["cat", "label", "n", "acc"] + bucket_names + ["wrong"]

    rows = []
    for cat in sorted(total):
        n = total[cat]
        c = correct[cat]
        wrong = n - c
        row = [
            str(cat),
            CATEGORY_LABELS.get(cat, f"cat_{cat}"),
            str(n),
            f"{c/n:.3f}" if n else "-",
        ]
        for b in bucket_names:
            row.append(str(buckets[cat].get(b, 0)))
        row.append(str(wrong))
        rows.append(row)

    # totals row
    n_total = sum(total.values())
    c_total = sum(correct.values())
    total_row = ["*", "OVERALL", str(n_total), f"{c_total/n_total:.3f}" if n_total else "-"]
    for b in bucket_names:
        total_row.append(str(sum(buckets[cat].get(b, 0) for cat in total)))
    total_row.append(str(n_total - c_total))
    rows.append(total_row)

    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line, file=file)
    print("-" * len(line), file=file)
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)), file=file)


def print_examples(result: dict, top: int, file=sys.stderr) -> None:
    print("\n=== EXAMPLES (top {} per bucket per category) ===".format(top), file=file)
    for (cat, bucket), recs in sorted(result["examples"].items()):
        if not recs:
            continue
        label = CATEGORY_LABELS.get(cat, f"cat_{cat}")
        print(f"\n[{label} / {bucket}]  ({len(recs)} cases)", file=file)
        for rec in recs[:top]:
            q = (rec.get("question") or "").strip()[:120]
            g = (rec.get("gold") or "").strip()[:120]
            p = (rec.get("pred") or "").strip()[:120]
            r5 = rec.get("r@5")
            print(f"  Q: {q}", file=file)
            print(f"  G: {g}", file=file)
            print(f"  P: {p}", file=file)
            print(f"     r@5={r5}  f1={rec.get('f1', 0):.2f}  rouge={rec.get('rouge_l', 0):.2f}", file=file)


def write_csv(classified: list[dict], path: Path) -> None:
    if not classified:
        return
    keys = list(classified[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in classified:
            row = {k: (v if not isinstance(v, str) else v.replace("\n", " ")) for k, v in row.items()}
            w.writerow(row)


def diff_runs(a: dict, b: dict) -> dict:
    """Return regressions (correct in A, wrong in B) and gains (wrong→correct)."""
    a_index = {(r.get("question"), r.get("gold")): r for r in a.get("records", [])}
    regressions: list[dict] = []
    gains: list[dict] = []
    for rb in b.get("records", []):
        key = (rb.get("question"), rb.get("gold"))
        ra = a_index.get(key)
        if ra is None:
            continue
        if ra.get("correct") and not rb.get("correct"):
            regressions.append({"question": key[0], "gold": key[1], "a_pred": ra.get("pred"), "b_pred": rb.get("pred"), "category": rb.get("category")})
        elif not ra.get("correct") and rb.get("correct"):
            gains.append({"question": key[0], "gold": key[1], "a_pred": ra.get("pred"), "b_pred": rb.get("pred"), "category": rb.get("category")})
    return {"regressions": regressions, "gains": gains}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("result", type=Path, help="Result JSON from locomo_bench_llm.py")
    p.add_argument("--csv", type=Path, help="Write classified rows to CSV")
    p.add_argument("--top", type=int, default=3, help="Examples per bucket per category to stderr")
    p.add_argument("--diff", type=Path, help="Compare against another result (regressions/gains)")
    p.add_argument("--quiet", action="store_true", help="Suppress example dump")
    args = p.parse_args(argv)

    with args.result.open() as f:
        data = json.load(f)
    records = data.get("records", [])
    if not records:
        print(f"no records in {args.result}", file=sys.stderr)
        return 2

    result = analyze(records)
    print(f"# Failure analysis: {args.result.name}")
    print(f"#   total={sum(result['by_cat_total'].values())} "
          f"correct={sum(result['by_cat_correct'].values())} "
          f"wrong={sum(c for cat in result['by_cat_bucket'] for c in result['by_cat_bucket'][cat].values())}")
    print()
    print_table(result)

    if not args.quiet:
        print_examples(result, top=args.top)

    if args.csv:
        write_csv(result["classified"], args.csv)
        print(f"\nCSV written: {args.csv} ({len(result['classified'])} rows)", file=sys.stderr)

    if args.diff:
        with args.diff.open() as f:
            data_b = json.load(f)
        d = diff_runs(data, data_b)
        print(f"\n=== DIFF vs {args.diff.name} ===")
        print(f"Regressions (A correct → B wrong): {len(d['regressions'])}")
        print(f"Gains       (A wrong   → B correct): {len(d['gains'])}")
        if d["regressions"]:
            print("\nRegressions sample (first 10):")
            for r in d["regressions"][:10]:
                print(f"  cat={r['category']}  Q: {(r['question'] or '')[:80]}")
                print(f"     gold: {(r['gold'] or '')[:80]}")
                print(f"     A:    {(r['a_pred'] or '')[:80]}")
                print(f"     B:    {(r['b_pred'] or '')[:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
