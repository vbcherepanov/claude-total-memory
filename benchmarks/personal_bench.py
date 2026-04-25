#!/usr/bin/env python3
"""
Personal-bench — health check of the user's REAL memory store.

Methodology:
  1. Sample N clean knowledge records from ~/.claude-memory (solutions,
     decisions, conventions, lessons, facts of decent length).
  2. Ask Haiku 4.5 to synthesize one Q&A pair per record where the answer
     is grounded in that record's content.
  3. For each synthesized question, run Recall.search() against the
     *real* memory DB.
  4. Metric: Source-Grounded Recall@K — was the source record returned
     in top-K results? Plus cross-check via LLM-judge (does the content
     of the top-1 result actually answer the question?).

Output: per-project recall scores + overall health score.

Safe by default: this script is READ-ONLY on the real DB — it only
calls Recall.search, never saves.

Usage:
    python benchmarks/personal_bench.py --n 150
    python benchmarks/personal_bench.py --n 300 --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RETRIEVAL_LOCK = threading.Lock()


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
REAL_DB = Path(os.path.expanduser("~/.claude-memory"))
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def setup_env() -> None:
    os.environ["CLAUDE_MEMORY_DIR"] = str(REAL_DB)
    os.environ.setdefault("MEMORY_LLM_ENABLED", "false")  # no writes; no Ollama calls


def import_store():
    sys.path.insert(0, "/Users/vitalii-macpro/claude-memory-server/src")
    import server
    return server


def patch_thread_safety(server_mod, store) -> None:
    import sqlite3
    try:
        store.db.close()
    except Exception:
        pass
    store.db = sqlite3.connect(str(server_mod.MEMORY_DIR / "memory.db"),
                               check_same_thread=False)
    store.db.row_factory = sqlite3.Row
    store.db.execute("PRAGMA journal_mode=WAL")
    store.db.execute("PRAGMA busy_timeout=5000")


QA_SYSTEM = """You generate high-quality retrieval-benchmark questions.
You will be given ONE knowledge snippet. Produce a single question+answer pair such that:
  - the question is specific (not "what is this about?") — e.g. about a concrete config, file path, command, name, date, reason, or decision recorded in the snippet;
  - the answer is a short phrase (<=20 words), directly supported by the snippet;
  - the question should be answerable ONLY if someone has access to this snippet (avoid trivially general questions);
  - prefer questions in the same language as the snippet.

Output STRICT JSON on a single line: {"question": "...", "answer": "..."}. No prose, no markdown fences."""


JUDGE_SYSTEM = """You check if a retrieved knowledge snippet genuinely answers a question.
Return YES if the snippet contains the information needed to answer the question, otherwise NO.
Respond with ONLY YES or NO on the first line."""


def call_haiku(client, system: str, user: str, max_tokens: int = 120,
               retries: int = 3) -> tuple[str, int, int]:
    import anthropic
    last = None
    for attempt in range(retries):
        try:
            r = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = r.content[0].text.strip() if r.content else ""
            return text, r.usage.input_tokens, r.usage.output_tokens
        except Exception as e:
            last = e
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"Haiku failed: {last}")


def sample_records(server_mod, n: int) -> list[dict]:
    store = server_mod.Store()
    rows = store.db.execute("""
        SELECT id, project, type, content, context, tags, created_at
        FROM knowledge
        WHERE status='active'
          AND length(content) BETWEEN 200 AND 2000
          AND type IN ('solution','decision','convention','lesson','fact')
          AND content NOT LIKE '%Session recovery context%'
          AND content NOT LIKE '%Session task:%'
          AND content NOT LIKE 'Last user messages%'
          AND project NOT IN ('tmp','')
        ORDER BY random()
        LIMIT ?
    """, (n,)).fetchall()
    return [dict(r) for r in rows]


def synth_qa(client, rec: dict) -> dict | None:
    snippet = rec["content"]
    prompt = (
        f"Project: {rec['project']}\n"
        f"Type: {rec['type']}\n"
        f"Snippet:\n{snippet}\n\n"
        f"Produce the JSON {{question, answer}} now."
    )
    try:
        text, tin, tout = call_haiku(client, QA_SYSTEM, prompt, max_tokens=180)
    except Exception as e:
        return None
    # Parse JSON (tolerant of leading/trailing noise)
    s = text.strip()
    # Strip markdown fences if Haiku added them despite instructions
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
    try:
        obj = json.loads(s)
        q = obj.get("question", "").strip()
        a = obj.get("answer", "").strip()
        if not q or not a:
            return None
        return {"question": q, "answer": a, "tokens_in": tin, "tokens_out": tout}
    except Exception:
        return None


def judge_snippet(client, question: str, snippet: str) -> tuple[bool, int, int]:
    prompt = (
        f"Question: {question}\n\nSnippet:\n{snippet[:1500]}\n\n"
        f"Does this snippet answer the question? YES or NO."
    )
    text, tin, tout = call_haiku(client, JUDGE_SYSTEM, prompt, max_tokens=4)
    return text.upper().startswith("YES"), tin, tout


def eval_one(client, server_mod, recall, rec: dict, top_k: int) -> dict | None:
    qa = synth_qa(client, rec)
    if qa is None:
        return None
    question = qa["question"]
    src_id = rec["id"]

    t0 = time.time()
    with RETRIEVAL_LOCK:
        res = recall.search(query=question, project=rec["project"],
                            limit=top_k, detail="summary")
    retr_ms = (time.time() - t0) * 1000
    entries = res.get("results", {})
    flat = []
    for typ, group in entries.items():
        flat.extend(group)
    flat.sort(key=lambda e: -e.get("score", 0))
    ids = [e["id"] for e in flat]

    # Source-grounded recall
    r1 = int(src_id in ids[:1])
    r5 = int(src_id in ids[:5])
    r10 = int(src_id in ids[:10])

    # Judge: does top-1 (regardless of id) actually answer?
    judge_ok = False
    j_in = j_out = 0
    if flat:
        try:
            judge_ok, j_in, j_out = judge_snippet(client, question, flat[0].get("content", ""))
        except Exception:
            judge_ok = False

    return {
        "src_id": src_id,
        "project": rec["project"],
        "type": rec["type"],
        "question": question,
        "answer": qa["answer"],
        "retrieved_ids": ids,
        "r@1_src": r1, "r@5_src": r5, "r@10_src": r10,
        "top1_answers": int(judge_ok),
        "retrieval_ms": retr_ms,
        "tokens_in": qa["tokens_in"] + j_in,
        "tokens_out": qa["tokens_out"] + j_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="Sample size")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    setup_env()
    server_mod = import_store()

    import anthropic
    client = anthropic.Anthropic()

    print(f"[personal-bench] sampling {args.n} records from {REAL_DB}")
    recs = sample_records(server_mod, args.n)
    # Shared Store+Recall; warm up lazy inits then patch thread safety
    shared_store = server_mod.Store()
    shared_recall = server_mod.Recall(shared_store)
    shared_recall.search(query="warmup", project="general", limit=1, detail="summary")
    patch_thread_safety(server_mod, shared_store)
    print(f"[personal-bench] got {len(recs)} records across "
          f"{len(set(r['project'] for r in recs))} projects")

    results: list[dict] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(eval_one, client, server_mod, shared_recall, r, args.top_k): r for r in recs}
        done = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"  eval error: {e}", file=sys.stderr)
            done += 1
            if done % 25 == 0 or done == len(recs):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.01)
                eta = (len(recs) - done) / max(rate, 0.01)
                if results:
                    r1 = sum(r["r@1_src"] for r in results) / len(results)
                    r5 = sum(r["r@5_src"] for r in results) / len(results)
                    r10 = sum(r["r@10_src"] for r in results) / len(results)
                    tj = sum(r["top1_answers"] for r in results) / len(results)
                    print(f"  [{done}/{len(recs)}] "
                          f"{rate:.1f} q/s eta={eta:.0f}s  "
                          f"R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} "
                          f"top1-answers={tj:.3f}", flush=True)

    # Aggregate
    per_project = defaultdict(list)
    for r in results:
        per_project[r["project"]].append(r)

    agg_projects = {}
    for p, rs in per_project.items():
        n = len(rs)
        agg_projects[p] = {
            "n": n,
            "R@1_src": round(sum(r["r@1_src"] for r in rs) / n, 4),
            "R@5_src": round(sum(r["r@5_src"] for r in rs) / n, 4),
            "R@10_src": round(sum(r["r@10_src"] for r in rs) / n, 4),
            "top1_answers": round(sum(r["top1_answers"] for r in rs) / n, 4),
        }

    overall = {
        "n": len(results),
        "R@1_src": round(sum(r["r@1_src"] for r in results) / max(len(results), 1), 4),
        "R@5_src": round(sum(r["r@5_src"] for r in results) / max(len(results), 1), 4),
        "R@10_src": round(sum(r["r@10_src"] for r in results) / max(len(results), 1), 4),
        "top1_answers": round(sum(r["top1_answers"] for r in results) / max(len(results), 1), 4),
    }
    lats = [r["retrieval_ms"] for r in results]
    latency = {
        "p50_ms": round(statistics.median(lats), 2) if lats else 0,
        "p95_ms": round(sorted(lats)[int(0.95 * len(lats))], 2) if lats else 0,
        "mean_ms": round(sum(lats) / max(len(lats), 1), 2),
    }

    print("\n" + "=" * 76)
    print("  Personal-bench — your real ~/.claude-memory store")
    print("=" * 76)
    print()
    print("Per-project (sorted by N)")
    print(f"  {'project':<30}  {'N':>4}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'Ans@1':>6}")
    for p, d in sorted(agg_projects.items(), key=lambda x: -x[1]["n"])[:20]:
        print(f"  {p[:30]:<30}  {d['n']:>4}  {d['R@1_src']:>6.3f}  "
              f"{d['R@5_src']:>6.3f}  {d['R@10_src']:>6.3f}  {d['top1_answers']:>6.3f}")
    print()
    print(f"Overall  N={overall['n']}  "
          f"R@1-src={overall['R@1_src']}  R@5-src={overall['R@5_src']}  "
          f"R@10-src={overall['R@10_src']}  top1-answers={overall['top1_answers']}")
    print(f"Latency  p50={latency['p50_ms']} ms  p95={latency['p95_ms']} ms  mean={latency['mean_ms']} ms")
    tok_in = sum(r["tokens_in"] for r in results)
    tok_out = sum(r["tokens_out"] for r in results)
    print(f"Haiku tokens  in={tok_in:,} out={tok_out:,}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"personal-{int(time.time())}.json"
    with open(out_path, "w") as fh:
        json.dump({
            "overall": overall,
            "per_project": agg_projects,
            "latency": latency,
            "records": results,
            "config": {"n": args.n, "top_k": args.top_k, "concurrency": args.concurrency,
                       "model": HAIKU_MODEL},
        }, fh, indent=2, ensure_ascii=False)
    print(f"[personal-bench] report → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
