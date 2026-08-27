#!/usr/bin/env python3
"""BEAM retrieval benchmark — "Beyond a Million Tokens" (ICLR 2026).

    https://github.com/mohammadtavakoli78/BEAM
    https://huggingface.co/datasets/Mohammadta/BEAM   (CC BY-SA 4.0)

BEAM is the benchmark that pushes memory past the point where a context
window can absorb the transcript: conversations of 100K / 500K / 1M tokens
(a separate BEAM-10M set goes further), with probing questions spanning ten
memory abilities — information extraction, multi-hop reasoning over sessions,
knowledge update, temporal reasoning, event ordering, contradiction
resolution, summarization, instruction/preference following, and abstention.

What this runner measures
-------------------------
**Retrieval only.** Each probing question carries `source_chat_ids`: the chat
messages that actually contain the answer. We ingest every message tagged with
its id and ask whether the memory surfaces the gold messages in the top-K.
That is gradable with no LLM in the loop, so the numbers are deterministic,
free, and reproducible on any machine.

It is deliberately *not* the paper's end-to-end accuracy, which needs a
generator and an LLM judge. Answer quality is bounded above by retrieval, so
this measures the part this project owns.

`abstention` questions have no gold evidence by construction — the right
behaviour is to surface nothing convincing — so they are reported separately
as a mean top-score, where lower is better.

Usage
-----
    python benchmarks/beam_bench.py --scale 100K
    python benchmarks/beam_bench.py --scale 1M --limit-chats 5
    python benchmarks/beam_bench.py --scale 500K --skip-ingest   # reuse DB

Download the parquet shards into benchmarks/data/beam/ first; the directory
is gitignored because the corpus carries its own licence.
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
DATA_DIR = ROOT / "benchmarks" / "data" / "beam"
RESULTS_DIR = ROOT / "benchmarks" / "results"
SCALES = ("100K", "500K", "1M")
DEFAULT_DB = Path("/tmp/beam_bench_db")

# Reported separately: no gold evidence exists for these by design.
NO_EVIDENCE_ABILITY = "abstention"


def setup_env(db_path: Path) -> None:
    os.environ["CLAUDE_MEMORY_DIR"] = str(db_path)
    # Retrieval is what we measure; LLM triple extraction only slows ingest.
    os.environ["MEMORY_LLM_ENABLED"] = "false"
    os.environ.setdefault("MEMORY_QUIET", "1")


def import_store():
    sys.path.insert(0, str(ROOT / "src"))
    import server  # noqa: F401 — import triggers MEMORY_DIR resolution

    return server


def dataset_path(scale: str) -> Path:
    return DATA_DIR / f"{scale}-00000-of-00001.parquet"


def load_dataset(scale: str) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover — surfaced to the operator
        raise SystemExit(
            "pyarrow is required to read the BEAM shards: pip install pyarrow"
        )
    path = dataset_path(scale)
    if not path.is_file():
        raise SystemExit(
            f"missing {path}\nDownload it with:\n"
            f"  curl -L -o {path} \\\n"
            f"    https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/"
            f"data/{scale}-00000-of-00001.parquet"
        )
    return pq.read_table(path).to_pydict()


def parse_probes(raw: str) -> dict:
    """`probing_questions` is a Python repr, not JSON."""
    import ast

    return ast.literal_eval(raw)


def gold_ids(probe: dict) -> set[int]:
    """Flatten `source_chat_ids`, which is a list or a dict of lists."""
    raw = probe.get("source_chat_ids")
    out: set[int] = set()

    def absorb(value) -> None:
        if isinstance(value, int):
            out.add(value)
        elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
            out.add(int(value))
        elif isinstance(value, dict):
            for v in value.values():
                absorb(v)
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                absorb(v)

    absorb(raw)
    return out


def msg_tag(conv_id: str, msg_id: int) -> str:
    return f"beam{conv_id}:{msg_id}"


def extract_msg_tags(entry: dict) -> list[str]:
    """Pull the ingest tag back out of a search hit (tags are lowercased)."""
    tags = entry.get("tags") or []
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except ValueError:
            tags = []
    return [
        t.lower()
        for t in tags
        if isinstance(t, str) and t.lower().startswith("beam") and ":" in t
    ]


def ingest(server_mod, data: dict, limit_chats: int | None,
           progress_every: int = 500) -> dict:
    store = server_mod.Store()
    started = time.time()
    saved = skipped = 0
    n_chats = len(data["conversation_id"])
    if limit_chats:
        n_chats = min(n_chats, limit_chats)

    for i in range(n_chats):
        conv_id = str(data["conversation_id"][i])
        project = f"beam_{conv_id}"
        for session_idx, session in enumerate(data["chat"][i]):
            sid = f"{project}__s{session_idx}"
            store.session_start(sid, project=project)
            for message in session:
                content = (message.get("content") or "").strip()
                msg_id = message.get("id")
                if not content or msg_id is None:
                    skipped += 1
                    continue
                role = message.get("role") or "user"
                anchor = message.get("time_anchor") or ""
                # Speaker and date go into the text: retrieval has the same
                # cues a reader would, matching the paper's setup.
                text = f"[{role}] {content}"
                if anchor:
                    text = f"[{anchor}] {text}"
                try:
                    store.save_knowledge(
                        sid=sid,
                        content=text,
                        ktype="fact",
                        project=project,
                        tags=[msg_tag(conv_id, msg_id), role, f"s{session_idx}"],
                        context=f"beam conv={conv_id} session={session_idx} id={msg_id}",
                        skip_dedup=True,
                        skip_quality=True,
                    )
                    saved += 1
                except Exception as e:
                    skipped += 1
                    print(f"  [ingest] save error {conv_id}:{msg_id}: {e}",
                          file=sys.stderr)
                if saved and saved % progress_every == 0:
                    rate = saved / max(time.time() - started, 1e-6)
                    print(f"  [ingest] {saved} messages ({rate:.1f}/s, "
                          f"chat={i + 1}/{n_chats})", flush=True)

    elapsed = time.time() - started
    return {
        "chats": n_chats,
        "saved": saved,
        "skipped": skipped,
        "elapsed_sec": round(elapsed, 2),
        "rate_msg_per_sec": round(saved / max(elapsed, 1e-6), 2),
    }


def evaluate(server_mod, data: dict, limit_chats: int | None,
             top_k: int) -> dict:
    store = server_mod.Store()
    recall = server_mod.Recall(store)

    per_ability = defaultdict(
        lambda: {"n": 0, "r@1": 0, "r@5": 0, "r@10": 0, "first_rank": []}
    )
    latencies: list[float] = []
    abstention_top: list[float] = []

    n_chats = len(data["conversation_id"])
    if limit_chats:
        n_chats = min(n_chats, limit_chats)

    for i in range(n_chats):
        conv_id = str(data["conversation_id"][i])
        project = f"beam_{conv_id}"
        probes = parse_probes(data["probing_questions"][i])
        for ability, items in probes.items():
            for probe in items:
                question = (probe.get("question") or "").strip()
                if not question:
                    continue

                t0 = time.time()
                # record_usage=False — see locomo_bench: the recall counter
                # feeds back into scoring and would make each re-run measure
                # its own history.
                res = recall.search(query=question, project=project,
                                    limit=top_k, detail="summary",
                                    record_usage=False)
                latencies.append((time.time() - t0) * 1000)
                entries = res.get("results", {}).get("fact", [])

                if ability == NO_EVIDENCE_ABILITY:
                    abstention_top.append(entries[0]["score"] if entries else 0.0)
                    continue

                gold = {msg_tag(conv_id, mid).lower() for mid in gold_ids(probe)}
                if not gold:
                    continue

                ranked = [
                    (extract_msg_tags(e) or [""])[0] for e in entries
                ]
                bucket = per_ability[ability]
                bucket["n"] += 1
                for k in (1, 5, 10):
                    bucket[f"r@{k}"] += int(any(t in gold for t in ranked[:k]))
                for rank, tag in enumerate(ranked, start=1):
                    if tag in gold:
                        bucket["first_rank"].append(rank)
                        break

    abilities: dict[str, dict] = {}
    totals = defaultdict(int)
    all_ranks: list[int] = []
    for ability, d in sorted(per_ability.items()):
        n = d["n"]
        if not n:
            continue
        entry = {
            "n": n,
            "R@1": round(d["r@1"] / n, 4),
            "R@5": round(d["r@5"] / n, 4),
            "R@10": round(d["r@10"] / n, 4),
            "MRR": round(sum(1.0 / r for r in d["first_rank"]) / n, 4),
        }
        abilities[ability] = entry
        for key in ("n", "r@1", "r@5", "r@10"):
            totals[key] += d[key]
        all_ranks.extend(d["first_rank"])

    overall = {}
    if totals["n"]:
        overall = {
            "n": totals["n"],
            "R@1": round(totals["r@1"] / totals["n"], 4),
            "R@5": round(totals["r@5"] / totals["n"], 4),
            "R@10": round(totals["r@10"] / totals["n"], 4),
            "MRR": round(sum(1.0 / r for r in all_ranks) / totals["n"], 4),
        }

    latency = {
        "p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_ms": (round(sorted(latencies)[int(0.95 * len(latencies))], 2)
                   if latencies else 0.0),
        "mean_ms": round(sum(latencies) / max(len(latencies), 1), 2),
        "queries": len(latencies),
    }
    abstention = {
        "n": len(abstention_top),
        "mean_top_score": round(
            sum(abstention_top) / max(len(abstention_top), 1), 4),
        "note": "no gold evidence by design — lower is better",
    }
    return {"per_ability": abilities, "overall": overall,
            "latency": latency, "abstention": abstention}


def format_report(scale: str, ingest_stats: dict, eval_stats: dict) -> str:
    out = ["=" * 74,
           f"  BEAM Benchmark (retrieval) — scale {scale}",
           "=" * 74, "",
           "Ingestion",
           f"  chats        : {ingest_stats.get('chats', 0)}",
           f"  messages     : {ingest_stats.get('saved', 0)}",
           f"  skipped      : {ingest_stats.get('skipped', 0)}",
           f"  elapsed      : {ingest_stats.get('elapsed_sec', 0)} s",
           f"  rate         : {ingest_stats.get('rate_msg_per_sec', 0)} msg/s",
           "",
           "Retrieval by memory ability",
           f"  {'ability':<26} {'N':>5} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}"]
    for ability, d in eval_stats["per_ability"].items():
        out.append(f"  {ability:<26} {d['n']:>5} {d['R@1']:>7.3f} "
                   f"{d['R@5']:>7.3f} {d['R@10']:>7.3f} {d['MRR']:>7.3f}")
    o = eval_stats.get("overall") or {}
    if o:
        out.append(f"  {'OVERALL':<26} {o['n']:>5} {o['R@1']:>7.3f} "
                   f"{o['R@5']:>7.3f} {o['R@10']:>7.3f} {o['MRR']:>7.3f}")
    lat = eval_stats["latency"]
    out += ["",
            "Latency",
            f"  p50: {lat['p50_ms']} ms  p95: {lat['p95_ms']} ms  "
            f"mean: {lat['mean_ms']} ms  (N={lat['queries']})",
            "",
            "Abstention (no gold evidence)",
            f"  N={eval_stats['abstention']['n']}  "
            f"mean top-score={eval_stats['abstention']['mean_top_score']}  "
            f"(lower is better)",
            ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", choices=SCALES, default="100K")
    ap.add_argument("--db-path", type=Path, default=None)
    ap.add_argument("--limit-chats", type=int, default=None,
                    help="Only ingest/eval the first N conversations")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-ingest", action="store_true",
                    help="Reuse an existing DB (ingest already done)")
    ap.add_argument("--wipe", action="store_true",
                    help="Remove the DB dir before ingest")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    db_path = args.db_path or Path(f"{DEFAULT_DB}_{args.scale}")
    if args.wipe and db_path.exists():
        shutil.rmtree(db_path)

    setup_env(db_path)
    server_mod = import_store()
    data = load_dataset(args.scale)

    n_chats = len(data["conversation_id"])
    print(f"[beam] scale={args.scale} chats={n_chats}"
          f"{f' (limited to {args.limit_chats})' if args.limit_chats else ''}")

    if args.skip_ingest:
        ingest_stats: dict = {}
        print("[beam] ingest skipped")
    else:
        print(f"[beam] ingestion → {db_path}")
        ingest_stats = ingest(server_mod, data, args.limit_chats)

    print(f"[beam] eval (top_k={args.top_k})")
    eval_stats = evaluate(server_mod, data, args.limit_chats, args.top_k)

    report = format_report(args.scale, ingest_stats, eval_stats)
    print(report)

    out = args.output or (RESULTS_DIR / f"beam-{args.scale}-{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": "BEAM (retrieval)",
        "scale": args.scale,
        "dataset": f"Mohammadta/BEAM · {dataset_path(args.scale).name}",
        "ingest": ingest_stats,
        "eval": eval_stats,
        "config": {
            "db_path": str(db_path),
            "top_k": args.top_k,
            "limit_chats": args.limit_chats,
            "llm_enabled": False,
        },
    }, indent=2))
    print(f"[beam] report → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
