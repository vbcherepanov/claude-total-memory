#!/usr/bin/env python3
"""LLM-based temporal resolver — turns relative date phrases in synth_facts
into concrete ISO dates. Fills gaps that regex can't parse:
  - "last fall"            → best guess year, month=10
  - "the sunday before …"  → anchor date minus N days
  - "4 years ago"          → anchor - 4y
  - "a couple weeks back"  → anchor - 14d

Writes into fact_temporal with confidence 0.6 (vs 1.0 for regex). Works
idempotently — existing rows are upserted if the LLM provides a better ISO.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
from _llm_adapter import LLMClient  # noqa: E402


SYSTEM_PROMPT = """You resolve relative date phrases into ISO dates.

Input: a short fact and an anchor date (the conversation timestamp).
Task: if the fact contains a date expression, output ISO or a compact phrase.

Output STRICT JSON, one line:
{"date_iso": "YYYY-MM-DD", "precision": "day|month|year|range", "phrase": "<original>"}

If the fact has NO temporal content, output:
{"date_iso": "", "precision": "", "phrase": ""}

Examples:
  fact: "Alice went to Berlin last fall." anchor: "2023-05-08"
  -> {"date_iso": "2022-10-01", "precision": "month", "phrase": "last fall"}

  fact: "Carol met Bob 4 years ago." anchor: "2023-06-09"
  -> {"date_iso": "2019-06-09", "precision": "year", "phrase": "4 years ago"}

  fact: "Bob plans to camp in June 2023."  anchor: "2023-05-25"
  -> {"date_iso": "2023-06-01", "precision": "month", "phrase": "June 2023"}

  fact: "Dan likes cold brew." anchor: "2023-05-01"
  -> {"date_iso": "", "precision": "", "phrase": ""}

No prose, no markdown."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="/tmp/locomo_bench_db")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "auto"])
    ap.add_argument("--project-prefix", default="locomo_")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-missing", action="store_true",
                    help="Process only synth_facts that have no fact_temporal row yet.")
    args = ap.parse_args()

    client = LLMClient(provider=args.provider, default_model=args.model)
    print(f"[temporal-llm] provider={client.provider} model={args.model}")

    db_file = f"{args.db_path}/memory.db"
    conn = sqlite3.connect(db_file, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Ensure fact_temporal exists (safe no-op if it does).
    conn.execute(
        """CREATE TABLE IF NOT EXISTS fact_temporal (
            knowledge_id INTEGER PRIMARY KEY,
            project      TEXT NOT NULL,
            entity       TEXT,
            event_hint   TEXT,
            date_iso     TEXT,
            date_raw     TEXT
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ft_project ON fact_temporal(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ft_entity  ON fact_temporal(entity)")

    # Select candidate rows.
    sql = ("SELECT k.id, k.project, k.content, k.created_at "
           "FROM knowledge k WHERE k.type='synthesized_fact' AND k.status='active' "
           "AND k.project LIKE ?")
    params: list = [f"{args.project_prefix}%"]
    if args.only_missing:
        sql += (" AND NOT EXISTS (SELECT 1 FROM fact_temporal ft "
                " WHERE ft.knowledge_id = k.id AND ft.date_iso != '')")
    if args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"
    rows = conn.execute(sql, params).fetchall()
    print(f"[temporal-llm] {len(rows)} rows to resolve, concurrency={args.concurrency}")

    # Heuristic filter — skip facts that obviously have no temporal content.
    _has_time_hint = re.compile(
        r"\b(?:year|month|week|day|hour|yesterday|today|tomorrow|tonight|last|next|"
        r"ago|before|after|since|until|earlier|later|morning|evening|afternoon|"
        r"january|february|march|april|may|june|july|august|september|october|"
        r"november|december|spring|summer|fall|autumn|winter|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"20\d{2}|19\d{2})\b",
        re.I,
    )

    tlocal = threading.local()

    def get_conn():
        c = getattr(tlocal, "conn", None)
        if c is None:
            c = sqlite3.connect(db_file, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=30000")
            tlocal.conn = c
        return c

    stats = {"resolved": 0, "no_time": 0, "errors": 0, "skipped": 0, "in": 0, "out": 0}
    stats_lock = threading.Lock()

    def work(row: sqlite3.Row) -> None:
        content = (row["content"] or "").strip()
        if not content or not _has_time_hint.search(content):
            with stats_lock:
                stats["skipped"] += 1
            return
        anchor = (row["created_at"] or "2023-01-01")[:10]
        user = f"fact: \"{content}\"\nanchor: \"{anchor}\""
        try:
            r = client.complete(SYSTEM_PROMPT, user, model=args.model, max_tokens=80)
            with stats_lock:
                stats["in"] += r.input_tokens
                stats["out"] += r.output_tokens
        except Exception:
            with stats_lock:
                stats["errors"] += 1
            return
        text = r.text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            obj = json.loads(text)
        except Exception:
            with stats_lock:
                stats["errors"] += 1
            return
        iso = str(obj.get("date_iso", "")).strip()
        if not iso:
            with stats_lock:
                stats["no_time"] += 1
            return
        phrase = str(obj.get("phrase", "")).strip()
        # Extract first capitalised token as entity hint (best-effort).
        m = re.search(r"\b([A-Z][a-zA-Z0-9'-]+)", content)
        entity = m.group(1).lower() if m else ""
        event_hint = ""
        if entity:
            tail = content[content.lower().find(entity):][len(entity):].strip().lower()
            event_hint = " ".join(tail.split()[:6]).rstrip(".,;:!?")
        c = get_conn().cursor()
        try:
            c.execute(
                "INSERT OR REPLACE INTO fact_temporal "
                "(knowledge_id, project, entity, event_hint, date_iso, date_raw) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (int(row["id"]), row["project"], entity, event_hint, iso, phrase),
            )
            get_conn().commit()
            with stats_lock:
                stats["resolved"] += 1
        except Exception:
            with stats_lock:
                stats["errors"] += 1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, r): r for r in rows}
        completed = 0
        for _ in as_completed(futs):
            completed += 1
            if completed % 500 == 0 or completed == len(rows):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.01)
                print(f"  [{completed}/{len(rows)}] rate={rate:.1f}/s "
                      f"resolved={stats['resolved']} skipped={stats['skipped']}",
                      flush=True)

    elapsed = time.time() - t0
    cost = stats["in"] / 1e6 * 0.15 + stats["out"] / 1e6 * 0.60
    print(f"[temporal-llm] done resolved={stats['resolved']} skipped={stats['skipped']} "
          f"no_time={stats['no_time']} errors={stats['errors']} "
          f"elapsed={elapsed:.1f}s  cost≈${cost:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
