#!/usr/bin/env python3
"""
LoCoMo Benchmark Runner for Claude Total Memory v8.

Evaluates the production memory pipeline (Store.save_knowledge + Recall.search)
against the LoCoMo dataset (Snap Research / USC, ACL 2024).

Dataset:  benchmarks/data/locomo/data/locomo10.json
          10 conversations, 272 sessions, 5882 turns, 1986 QA pairs
          Categories: 1=single-hop, 2=multi-hop, 3=temporal,
                      4=open-domain, 5=adversarial

Metrics:
  - Recall@1, Recall@5, Recall@10 per category and overall
  - Mean rank of first gold evidence (MRR-style)
  - Retrieval latency p50/p95
  - Adversarial rejection rate (for category 5 — expect *no* strong match)

Usage:
    python benchmarks/locomo_bench.py                       # full run
    python benchmarks/locomo_bench.py --limit-samples 2     # first 2 convs
    python benchmarks/locomo_bench.py --limit-qa 100        # first 100 QAs
    python benchmarks/locomo_bench.py --skip-ingest         # reuse existing DB
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "benchmarks" / "data" / "locomo" / "data" / "locomo10.json"
DEFAULT_DB = Path("/tmp/locomo_bench_db")
RESULTS_DIR = ROOT / "benchmarks" / "results"


def setup_env(db_path: Path, disable_llm: bool) -> None:
    os.environ["CLAUDE_MEMORY_DIR"] = str(db_path)
    # Separate Chroma/embeddings dir implied via CLAUDE_MEMORY_DIR
    if disable_llm:
        # Keep embeddings (fastembed/ST); skip Ollama-based triple extraction
        # so ingestion finishes in reasonable time. Retrieval is unaffected.
        os.environ["MEMORY_LLM_ENABLED"] = "false"
    # Quiet the server's LOG prints so runner output stays readable
    os.environ.setdefault("MEMORY_QUIET", "1")


def import_store():
    sys.path.insert(0, str(Path("/Users/vitalii-macpro/claude-memory-server/src")))
    import server  # noqa: F401  — triggers MEMORY_DIR resolution
    return server


def load_dataset(path: Path) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


def ingest(server_mod, samples: list[dict], progress_every: int = 200) -> dict:
    """Save every turn as a knowledge fact. Returns counters."""
    store = server_mod.Store()
    t0 = time.time()
    saved = 0
    skipped = 0

    for sample_idx, sample in enumerate(samples):
        sample_id = sample.get("sample_id", f"conv_{sample_idx}")
        project = f"locomo_{sample_idx}"
        conv = sample["conversation"]
        speaker_a = conv.get("speaker_a", "A")
        speaker_b = conv.get("speaker_b", "B")

        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda k: int(k.split("_")[1]),
        )

        for sk in session_keys:
            sess_date = conv.get(f"{sk}_date_time", "")
            sid = f"{project}__{sk}"
            store.session_start(sid, project=project)

            turns = conv[sk]
            if not isinstance(turns, list):
                continue
            for turn in turns:
                dia_id = turn.get("dia_id", "")
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                if not text or not dia_id:
                    skipped += 1
                    continue
                # Include session timestamp + speaker in content so semantic
                # retrieval has temporal/speaker cues (matches paper's setup).
                content = f"[{sess_date}] {speaker}: {text}"
                tags = [dia_id, sk, "locomo", f"conv_{sample_idx}", f"speaker:{speaker}"]
                if turn.get("blip_caption"):
                    content += f"\n(image: {turn['blip_caption']})"
                try:
                    store.save_knowledge(
                        sid=sid,
                        content=content,
                        ktype="fact",
                        project=project,
                        tags=tags,
                        context=f"locomo conv={sample_id} session={sk} dia_id={dia_id}",
                        skip_dedup=True,
                    )
                    saved += 1
                    if saved % progress_every == 0:
                        elapsed = time.time() - t0
                        rate = saved / max(elapsed, 0.01)
                        print(f"  [ingest] {saved} turns saved ({rate:.1f} turn/s, "
                              f"conv={sample_idx+1}/{len(samples)})", flush=True)
                except Exception as e:
                    skipped += 1
                    print(f"  [ingest] save error dia_id={dia_id}: {e}", file=sys.stderr)

    # Commit any pending writes
    try:
        store.db.commit()
    except Exception:
        pass

    elapsed = time.time() - t0
    return {
        "saved": saved,
        "skipped": skipped,
        "elapsed_sec": round(elapsed, 2),
        "rate_turn_per_sec": round(saved / max(elapsed, 0.01), 2),
    }


def extract_dia_ids(entry: dict) -> list[str]:
    """Pull dia_id tags out of a Recall result entry."""
    tags = entry.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except Exception:
            tags = []
    return [t for t in tags if isinstance(t, str) and t.startswith("D") and ":" in t]


def eval_samples(server_mod, samples: list[dict], top_k: int = 10,
                 limit_qa: int | None = None) -> dict:
    store = server_mod.Store()
    recall = server_mod.Recall(store)

    per_cat = defaultdict(lambda: {"n": 0, "r@1": 0, "r@5": 0, "r@10": 0,
                                   "first_rank": [], "gold_in_topk_cnt": 0})
    latencies: list[float] = []
    adversarial_top_score: list[float] = []

    qa_count = 0
    for sample_idx, sample in enumerate(samples):
        project = f"locomo_{sample_idx}"
        for qa in sample.get("qa", []):
            if limit_qa and qa_count >= limit_qa:
                break
            cat = qa.get("category", 0)
            question = qa.get("question", "")
            evidence = set(qa.get("evidence", []) or [])

            t0 = time.time()
            res = recall.search(query=question, project=project, limit=top_k,
                                detail="summary")
            latencies.append((time.time() - t0) * 1000)

            # All entries are type="fact"
            entries = res.get("results", {}).get("fact", [])
            ranked_dia_ids: list[str] = []
            for entry in entries:
                dids = extract_dia_ids(entry)
                # An entry carries exactly one dia_id by construction; take first
                ranked_dia_ids.append(dids[0] if dids else "")

            qa_count += 1
            bucket = per_cat[cat]
            bucket["n"] += 1

            if cat == 5:
                # Adversarial: answer is "Not mentioned"; log top score
                top_score = entries[0]["score"] if entries else 0.0
                adversarial_top_score.append(top_score)
                continue

            if not evidence:
                # No gold evidence → cannot score recall, skip
                bucket["n"] -= 1
                continue

            hit_1 = any(d in evidence for d in ranked_dia_ids[:1])
            hit_5 = any(d in evidence for d in ranked_dia_ids[:5])
            hit_10 = any(d in evidence for d in ranked_dia_ids[:10])

            bucket["r@1"] += int(hit_1)
            bucket["r@5"] += int(hit_5)
            bucket["r@10"] += int(hit_10)

            first_rank = None
            for rank, did in enumerate(ranked_dia_ids, start=1):
                if did in evidence:
                    first_rank = rank
                    break
            if first_rank is not None:
                bucket["first_rank"].append(first_rank)
                bucket["gold_in_topk_cnt"] += 1
        else:
            continue
        break  # propagate limit_qa break

    # Aggregate
    agg = {}
    for cat, d in per_cat.items():
        n = d["n"]
        if n == 0:
            continue
        entry = {
            "n": n,
            "R@1": round(d["r@1"] / n, 4),
            "R@5": round(d["r@5"] / n, 4),
            "R@10": round(d["r@10"] / n, 4),
        }
        if d["first_rank"]:
            entry["MRR"] = round(sum(1.0 / r for r in d["first_rank"]) / n, 4)
            entry["mean_first_rank"] = round(
                sum(d["first_rank"]) / len(d["first_rank"]), 2)
            entry["found_in_top10"] = round(d["gold_in_topk_cnt"] / n, 4)
        else:
            entry["MRR"] = 0.0
            entry["found_in_top10"] = 0.0
        agg[f"category_{cat}"] = entry

    # Overall (exclude adversarial)
    total_n = sum(d["n"] for c, d in per_cat.items() if c != 5)
    if total_n:
        overall = {
            "n": total_n,
            "R@1": round(sum(d["r@1"] for c, d in per_cat.items() if c != 5) / total_n, 4),
            "R@5": round(sum(d["r@5"] for c, d in per_cat.items() if c != 5) / total_n, 4),
            "R@10": round(sum(d["r@10"] for c, d in per_cat.items() if c != 5) / total_n, 4),
        }
        agg["overall"] = overall

    latency = {
        "p50_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "p95_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 2) if latencies else 0,
        "mean_ms": round(sum(latencies) / max(len(latencies), 1), 2),
        "queries": len(latencies),
    }
    adv = {
        "n": len(adversarial_top_score),
        "mean_top_score": round(
            sum(adversarial_top_score) / max(len(adversarial_top_score), 1), 4),
        "note": "Lower is better — adversarial has no gold evidence",
    }

    return {"per_category": agg, "latency": latency, "adversarial": adv}


CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}


def format_report(ingest_stats: dict, eval_stats: dict) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append("  LoCoMo Benchmark — Claude Total Memory v8")
    lines.append("=" * 68)
    lines.append("")
    lines.append("Ingestion")
    lines.append(f"  saved turns  : {ingest_stats.get('saved', 0)}")
    lines.append(f"  skipped      : {ingest_stats.get('skipped', 0)}")
    lines.append(f"  elapsed      : {ingest_stats.get('elapsed_sec', 0)} s")
    lines.append(f"  rate         : {ingest_stats.get('rate_turn_per_sec', 0)} turn/s")
    lines.append("")
    lines.append("Retrieval metrics")
    lines.append(f"  {'category':<20}  {'N':>5}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'MRR':>6}  {'MR1':>6}")
    per = eval_stats.get("per_category", {})
    for key in ("category_1", "category_2", "category_3", "category_4", "overall"):
        if key not in per:
            continue
        d = per[key]
        label = CATEGORY_NAMES.get(int(key.split("_")[1]), "overall") if key.startswith("category_") else "overall"
        label_full = f"{key} ({label})" if key.startswith("category_") else label
        lines.append(
            f"  {label_full:<20}  {d['n']:>5}  {d['R@1']:>6.3f}  {d['R@5']:>6.3f}  "
            f"{d['R@10']:>6.3f}  {d.get('MRR', 0):>6.3f}  {d.get('mean_first_rank', 0):>6.2f}"
        )
    lat = eval_stats.get("latency", {})
    adv = eval_stats.get("adversarial", {})
    lines.append("")
    lines.append("Latency")
    lines.append(f"  p50: {lat.get('p50_ms', 0)} ms  p95: {lat.get('p95_ms', 0)} ms  "
                 f"mean: {lat.get('mean_ms', 0)} ms  (N={lat.get('queries', 0)})")
    lines.append("")
    lines.append("Adversarial (category 5)")
    lines.append(f"  N={adv.get('n', 0)}  mean top-score={adv.get('mean_top_score', 0)}  "
                 f"(lower is better)")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Only ingest/eval first N conversations")
    parser.add_argument("--limit-qa", type=int, default=None,
                        help="Cap total QA queries")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Reuse existing DB (assumes ingest has been run)")
    parser.add_argument("--wipe", action="store_true",
                        help="Remove DB dir before ingest")
    parser.add_argument("--enable-llm", action="store_true",
                        help="Keep Ollama triple extraction (slow)")
    parser.add_argument("--output", default=None,
                        help="Write JSON report here; default = benchmarks/results/locomo-<ts>.json")
    args = parser.parse_args()

    dataset = load_dataset(DATASET)
    if args.limit_samples:
        dataset = dataset[: args.limit_samples]
    print(f"[locomo] samples={len(dataset)}  total QA={sum(len(s['qa']) for s in dataset)}")

    db_path = Path(args.db_path)
    if args.wipe and db_path.exists():
        print(f"[locomo] wiping {db_path}")
        shutil.rmtree(db_path, ignore_errors=True)

    setup_env(db_path, disable_llm=not args.enable_llm)
    server_mod = import_store()

    ingest_stats = {}
    if not args.skip_ingest:
        print(f"[locomo] ingestion → {db_path}")
        ingest_stats = ingest(server_mod, dataset)
    else:
        print("[locomo] skipping ingestion (reusing DB)")

    print(f"[locomo] eval (top_k={args.top_k}, limit_qa={args.limit_qa})")
    eval_stats = eval_samples(server_mod, dataset, top_k=args.top_k,
                              limit_qa=args.limit_qa)

    report = format_report(ingest_stats, eval_stats)
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"locomo-{int(time.time())}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "ingest": ingest_stats,
            "eval": eval_stats,
            "config": {
                "db_path": str(db_path),
                "top_k": args.top_k,
                "samples": len(dataset),
                "llm_enabled": args.enable_llm,
            },
        }, fh, indent=2)
    print(f"[locomo] report → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
