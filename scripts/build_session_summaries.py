#!/usr/bin/env python3
"""Generate one LLM summary per LoCoMo conversation and save as a
knowledge row (type='session_summary'). Retrieval then injects the matching
summary into the prompt context for richer open-domain answering.

Usage:
    python scripts/build_session_summaries.py \
        --db-path /tmp/locomo_bench_db \
        --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
from _llm_adapter import LLMClient  # noqa: E402


SYSTEM_PROMPT = """You summarise a long multi-turn conversation between two people.

Output one paragraph (6-12 lines, <1200 chars). Cover:
- Who the two participants are (names, relationships, roles).
- The recurring topics they discuss and major life events mentioned.
- Key factual anchors: dates, places, occupations, purchases, family members.
- Named entities (people, companies, cities) introduced.

Do NOT fabricate. Only use information that appears in the turns below."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="/tmp/locomo_bench_db")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "auto"])
    ap.add_argument("--project-prefix", default="locomo_")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--max-input-chars", type=int, default=60000,
                    help="Truncate conversation to this many chars (keep head+tail).")
    args = ap.parse_args()

    client = LLMClient(provider=args.provider, default_model=args.model)
    print(f"[summaries] provider={client.provider} model={args.model}")

    db_file = f"{args.db_path}/memory.db"
    conn = sqlite3.connect(db_file, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    if args.reset:
        n = conn.execute(
            "DELETE FROM knowledge WHERE type='session_summary' AND project LIKE ?",
            (f"{args.project_prefix}%",),
        ).rowcount
        conn.commit()
        print(f"[summaries] reset: {n} existing summaries deleted")

    # Discover distinct projects that look like LoCoMo conversations.
    projects = [r[0] for r in conn.execute(
        "SELECT DISTINCT project FROM knowledge WHERE project LIKE ? ORDER BY project",
        (f"{args.project_prefix}%",),
    ).fetchall()]
    print(f"[summaries] {len(projects)} conversations found")

    total_in = total_out = 0
    t0 = time.time()
    for project in projects:
        existing = conn.execute(
            "SELECT id FROM knowledge WHERE project=? AND type='session_summary' LIMIT 1",
            (project,),
        ).fetchone()
        if existing:
            print(f"  {project} — already summarised ({existing['id']}), skip")
            continue

        rows = conn.execute(
            "SELECT content FROM knowledge WHERE project=? AND type='fact' "
            "AND status='active' ORDER BY id ASC",
            (project,),
        ).fetchall()
        if not rows:
            continue

        joined = "\n".join(f"- {r['content']}" for r in rows if r['content'])
        # Head+tail truncation keeps intro (participants) + closing topics.
        if len(joined) > args.max_input_chars:
            half = args.max_input_chars // 2
            joined = joined[:half] + "\n[...]\n" + joined[-half:]

        try:
            r = client.complete(
                SYSTEM_PROMPT, f"Conversation turns:\n{joined}",
                model=args.model, max_tokens=450,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {project} — LLM err: {e}", file=sys.stderr)
            continue

        summary = r.text.strip()
        if not summary:
            continue

        conn.execute(
            "INSERT INTO knowledge "
            "(session_id, project, type, content, context, tags, confidence, "
            " created_at, last_confirmed, recall_count, status, branch) "
            "VALUES (?, ?, 'session_summary', ?, '', ?, 1.0, "
            " strftime('%Y-%m-%dT%H:%M:%SZ','now'), "
            " strftime('%Y-%m-%dT%H:%M:%SZ','now'), 0, 'active', '')",
            ("", project, summary, json.dumps(["session_summary", "locomo"])),
        )
        conn.commit()
        total_in += r.input_tokens
        total_out += r.output_tokens
        print(f"  {project} — {len(summary)} chars ({len(rows)} turns)  "
              f"tok in={r.input_tokens} out={r.output_tokens}")

    elapsed = time.time() - t0
    cost = total_in / 1e6 * 0.15 + total_out / 1e6 * 0.60
    print(f"[summaries] done {len(projects)} convs in {elapsed:.1f}s  "
          f"tok={total_in}→{total_out}  cost≈${cost:.3f}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
