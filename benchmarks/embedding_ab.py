#!/usr/bin/env python3
"""v9.0 B1 — A/B bench for embedding backends on LongMemEval subset.

For each selected backend (fastembed/minilm/e5-large/bge-m3) runs retrieval
against a fixed sample of LongMemEval questions, computes R@1, R@5 and mean
per-query latency, and dumps results to ``evals/embedding-ab-<date>.json``.

Example:
    python benchmarks/embedding_ab.py --backends fastembed,e5-large,bge-m3 \\
        --sample 50 --download

    python benchmarks/embedding_ab.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import choose_embed  # noqa: E402


DEFAULT_DATA = ROOT / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
DEFAULT_EVAL_DIR = ROOT / "evals"
DEFAULT_BACKENDS = ("fastembed", "e5-large", "bge-m3")
DEFAULT_SAMPLE = 50


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────


def _load_questions(path: Path, sample: int, stratified: bool, seed: int) -> list[dict]:
    with open(path) as fh:
        raw = json.load(fh)
    # Drop abstention questions — they expect no-answer behaviour.
    clean = [e for e in raw if not e.get("question_id", "").endswith("_abs")]
    if sample <= 0 or sample >= len(clean):
        return clean
    if not stratified:
        return clean[:sample]

    # Stratified sample by question_type to keep proportions roughly stable.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for e in clean:
        buckets[e.get("question_type", "_unknown")].append(e)

    total = len(clean)
    out: list[dict] = []
    for qtype, rows in buckets.items():
        share = max(1, round(sample * len(rows) / total))
        out.extend(rows[:share])
    # Trim/fill to exactly `sample`.
    if len(out) > sample:
        out = out[:sample]
    elif len(out) < sample:
        seen = {e["question_id"] for e in out}
        for e in clean:
            if e["question_id"] not in seen:
                out.append(e)
                if len(out) >= sample:
                    break
    return out


def _build_corpus(entry: dict) -> tuple[list[str], list[str]]:
    session_ids = entry["haystack_session_ids"]
    sessions = entry["haystack_sessions"]
    texts: list[str] = []
    ids: list[str] = []
    for sid, session in zip(session_ids, sessions):
        user_text = "\n".join(t["content"] for t in session if t["role"] == "user")
        if user_text.strip():
            texts.append(user_text)
            ids.append(sid)
    return texts, ids


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────


def _cosine_top_k(query_vec: list[float], corpus_vecs: list[list[float]], k: int) -> list[int]:
    """Top-k indices by cosine similarity (pure stdlib)."""
    import math

    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def norm(v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v)) or 1e-9

    q_norm = norm(query_vec)
    sims = []
    for i, c in enumerate(corpus_vecs):
        sims.append((i, dot(query_vec, c) / (q_norm * norm(c))))
    sims.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in sims[:k]]


def _recall_any(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    return float(any(r in gold_set for r in retrieved))


# ──────────────────────────────────────────────
# Benchmark per backend
# ──────────────────────────────────────────────


def _run_backend(backend: str, questions: list[dict], k_list: list[int]) -> dict:
    model_name = choose_embed.resolve_model_name(backend)
    print(f"\n[ab] backend={backend}  model={model_name}")
    provider = choose_embed.get_provider(backend)
    if not provider.available():
        print(f"[ab]   -> provider unavailable, skipping.")
        return {
            "backend": backend,
            "model": model_name,
            "available": False,
            "recall": {f"r@{k}": None for k in k_list},
            "latency_ms": {"mean": None, "p50": None, "p95": None},
            "n": 0,
        }

    max_k = max(k_list)
    results = {f"r@{k}": [] for k in k_list}
    latencies: list[float] = []

    for qi, entry in enumerate(questions):
        corpus_texts, corpus_ids = _build_corpus(entry)
        if not corpus_texts:
            continue
        question = entry["question"]
        gold_ids = entry["answer_session_ids"]

        t0 = time.time()
        # Batch embed: corpus first, then the query last to reuse warm model.
        all_vecs = provider.embed(corpus_texts + [question])
        corpus_vecs = all_vecs[:-1]
        q_vec = all_vecs[-1]
        top_indices = _cosine_top_k(q_vec, corpus_vecs, max_k)
        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)

        retrieved_ids = [corpus_ids[i] for i in top_indices]
        for k in k_list:
            results[f"r@{k}"].append(_recall_any(retrieved_ids[:k], gold_ids))

        if (qi + 1) % 10 == 0 or qi == len(questions) - 1:
            running = {
                key: (sum(v) / len(v) * 100.0 if v else 0.0) for key, v in results.items()
            }
            running_str = ", ".join(f"{k}={v:.1f}%" for k, v in running.items())
            print(f"[ab]   [{qi+1}/{len(questions)}] {running_str}")

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        srt = sorted(xs)
        idx = min(len(srt) - 1, int(len(srt) * p))
        return srt[idx]

    summary = {
        "backend": backend,
        "model": model_name,
        "available": True,
        "recall": {
            f"r@{k}": (_mean(results[f"r@{k}"]) if results[f"r@{k}"] else 0.0)
            for k in k_list
        },
        "latency_ms": {
            "mean": _mean(latencies),
            "p50": _pct(latencies, 0.50),
            "p95": _pct(latencies, 0.95),
        },
        "n": len(latencies),
    }
    return summary


# ──────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────


def _print_table(summaries: list[dict], k_list: list[int]) -> None:
    print("\n" + "=" * 78)
    print("v9.0 B1 Embedding A/B  —  LongMemEval subset")
    print("=" * 78)

    header = f"{'backend':<12} {'model':<52}"
    for k in k_list:
        header += f" {'R@'+str(k):>6}"
    header += f" {'ms/q':>7}"
    print(header)
    print("-" * len(header))

    for s in summaries:
        if not s["available"]:
            print(f"{s['backend']:<12} {s['model']:<52} {'(unavailable)':>40}")
            continue
        row = f"{s['backend']:<12} {s['model'][:52]:<52}"
        for k in k_list:
            row += f" {s['recall']['r@'+str(k)] * 100:>5.1f}%"
        row += f" {s['latency_ms']['mean']:>7.1f}"
        print(row)

    print("=" * 78)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A/B bench for v9 embedding backends.")
    p.add_argument(
        "--backends",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help=(
            "Comma-separated list of backends to benchmark. Supported: "
            "fastembed, minilm, e5-large, bge-m3."
        ),
    )
    p.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Number of questions (default: {DEFAULT_SAMPLE}, 0 = all).",
    )
    p.add_argument(
        "--stratified",
        action="store_true",
        help="Stratify the sample across question_type buckets.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed (reserved — sample currently deterministic by order).",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help=f"LongMemEval dataset (default: {DEFAULT_DATA}).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Default: evals/embedding-ab-<date>.json",
    )
    p.add_argument(
        "--k",
        type=str,
        default="1,5",
        help="Comma-separated K values (default: 1,5).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help=(
            "Allow the selected backends to download models (BGE-M3 ~2GB, "
            "e5-large ~1GB). Without it the run still proceeds, but any missing "
            "model will be reported as unavailable rather than downloaded."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    backends = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in backends if b not in choose_embed.BACKEND_MODEL]
    if unknown:
        print(f"[ab] ERROR: unknown backend(s): {unknown}", file=sys.stderr)
        print(f"[ab] supported: {list(choose_embed.BACKEND_MODEL)}", file=sys.stderr)
        return 2

    k_list = sorted({int(x) for x in args.k.split(",") if x.strip()})
    if not k_list:
        print("[ab] ERROR: --k must contain at least one value", file=sys.stderr)
        return 2

    if not args.data.exists():
        print(f"[ab] ERROR: dataset not found: {args.data}", file=sys.stderr)
        return 1

    if not args.download:
        print(
            "[ab] --download not set. Backends whose models are not already cached "
            "will appear as unavailable."
        )
        # Hint for HF/fastembed to avoid downloads of models that aren't there.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    questions = _load_questions(args.data, args.sample, args.stratified, args.seed)
    print(f"[ab] dataset: {args.data}")
    print(f"[ab] sample:  {len(questions)} questions (stratified={args.stratified})")
    type_counts = Counter(q.get("question_type", "_unknown") for q in questions)
    for qt, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"[ab]   - {qt}: {n}")
    print(f"[ab] backends: {backends}")
    print(f"[ab] K values: {k_list}")

    summaries = [_run_backend(b, questions, k_list) for b in backends]
    _print_table(summaries, k_list)

    out_path = args.out or (DEFAULT_EVAL_DIR / f"embedding-ab-{date.today().isoformat()}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date.today().isoformat(),
        "dataset": str(args.data),
        "sample": len(questions),
        "stratified": args.stratified,
        "k": k_list,
        "backends": summaries,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[ab] results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
