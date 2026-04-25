#!/usr/bin/env python3
"""Parallel drain of triple_extraction_queue via Anthropic Haiku.

Alternative to the queue-sequential reflection runner. Takes ~5 min for
5 882 items at concurrency=16 vs ~6.5h sequentially.

Only processes triple_extraction_queue (biggest ROI for retrieval). Skips
deep_enrichment and representations for now — add later if needed.

Usage:
    ANTHROPIC_API_KEY=... python benchmarks/parallel_drain.py --concurrency 16
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def setup_env(db_path: str) -> None:
    os.environ["CLAUDE_MEMORY_DIR"] = db_path
    os.environ["MEMORY_LLM_PROVIDER"] = "anthropic"
    os.environ["MEMORY_LLM_MODEL"] = "claude-haiku-4-5-20251001"
    os.environ["MEMORY_LLM_ENABLED"] = "true"
    os.environ["MEMORY_TRIPLE_PROVIDER"] = "anthropic"
    os.environ["MEMORY_TRIPLE_MODEL"] = "claude-haiku-4-5-20251001"
    # Propagate ANTHROPIC_API_KEY → MEMORY_LLM_API_KEY if only one is set
    if os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("MEMORY_LLM_API_KEY"):
        os.environ["MEMORY_LLM_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]


def import_memory_mods():
    sys.path.insert(0, "/Users/vitalii-macpro/claude-memory-server/src")
    from triple_extraction_queue import TripleExtractionQueue
    from ingestion.extractor import ConceptExtractor
    return TripleExtractionQueue, ConceptExtractor


def drain(db_path: str, concurrency: int = 16, limit: int = 0) -> dict:
    TripleExtractionQueue, ConceptExtractor = import_memory_mods()

    # Enqueue all active knowledge rows that aren't in the queue yet
    # (fresh ingests already enqueued; re-enqueue is idempotent INSERT OR IGNORE).
    main_db = sqlite3.connect(f"{db_path}/memory.db", check_same_thread=False)
    main_db.row_factory = sqlite3.Row
    main_db.execute("PRAGMA journal_mode=WAL")
    main_db.execute("PRAGMA busy_timeout=10000")

    # Enqueue everything active.
    q = TripleExtractionQueue(main_db)
    active_ids = [r[0] for r in main_db.execute(
        "SELECT id FROM knowledge WHERE status='active'").fetchall()]
    if limit > 0:
        active_ids = active_ids[:limit]
    for kid in active_ids:
        q.enqueue(kid)
    main_db.commit()

    pending = main_db.execute(
        "SELECT COUNT(*) FROM triple_extraction_queue WHERE status='pending'"
    ).fetchone()[0]
    print(f"[drain] {len(active_ids)} active knowledge rows, {pending} pending in queue")

    # Worker uses its own SQLite connection (check_same_thread=False on main, but
    # each thread gets its own connection via factory — safer).
    def process_one(kid: int) -> tuple[int, str]:
        t = threading.current_thread()
        conn = getattr(t, "_conn", None)
        if conn is None:
            conn = sqlite3.connect(f"{db_path}/memory.db", check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            t._conn = conn
            t._extractor = ConceptExtractor(conn)
        extractor = t._extractor
        content_row = conn.execute(
            "SELECT content FROM knowledge WHERE id=?", (kid,)
        ).fetchone()
        if content_row is None:
            return kid, "skip"
        try:
            extractor.extract_and_link(
                text=content_row["content"] or "", knowledge_id=kid, deep=True
            )
            # Mark queue row as done
            conn.execute(
                "UPDATE triple_extraction_queue SET status='done', processed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE knowledge_id=? AND status IN ('pending','processing')",
                (kid,),
            )
            conn.commit()
            return kid, "ok"
        except Exception as e:
            conn.execute(
                "UPDATE triple_extraction_queue SET status='failed', last_error=?, attempts=attempts+1 "
                "WHERE knowledge_id=? AND status IN ('pending','processing')",
                (str(e)[:500], kid),
            )
            conn.commit()
            return kid, f"err: {e!s}"

    # Fetch pending kids
    rows = main_db.execute(
        "SELECT knowledge_id FROM triple_extraction_queue WHERE status='pending'"
    ).fetchall()
    pending_ids = [r[0] for r in rows]
    print(f"[drain] {len(pending_ids)} to process  concurrency={concurrency}")

    t0 = time.time()
    done = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(process_one, kid): kid for kid in pending_ids}
        completed = 0
        for fut in as_completed(futures):
            kid, status = fut.result()
            if status == "ok":
                done += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
            completed += 1
            if completed % 100 == 0 or completed == len(pending_ids):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.01)
                eta = (len(pending_ids) - completed) / max(rate, 0.01)
                print(f"  [{completed}/{len(pending_ids)}] done={done} failed={failed} "
                      f"rate={rate:.2f}/s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    return {
        "total": len(pending_ids),
        "done": done, "failed": failed, "skipped": skipped,
        "elapsed_sec": round(elapsed, 2),
        "rate_per_sec": round(len(pending_ids) / max(elapsed, 0.01), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="/tmp/locomo_bench_db")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N knowledge rows (0 = all)")
    args = ap.parse_args()

    setup_env(args.db_path)
    stats = drain(args.db_path, concurrency=args.concurrency, limit=args.limit)
    print(f"[drain] final: {stats}")
    return 0


if __name__ == "__main__":
    import threading
    sys.exit(main())
