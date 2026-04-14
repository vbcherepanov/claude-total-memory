"""Backfill connections for orphan graph nodes via Ollama deep extraction.

Finds graph_nodes with zero edges (orphans), traces them back to their
source knowledge records, and enqueues those records into the
triple_extraction_queue. The reflection runner (LaunchAgent) will drain the
queue and Ollama will extract (subject, predicate, object) triples —
creating real graph_edges where there were none.

Usage:
    ~/claude-memory-server/.venv/bin/python src/tools/backfill_orphan_edges.py [--limit=N] [--min-mentions=N]

Options:
    --limit=N         — Process at most N orphan nodes (default: 200)
    --min-mentions=N  — Only backfill orphans with >= N mentions (default: 2).
                        Rare one-off concepts rarely yield useful triples.
    --trigger-now     — Touch .reflect-pending so the runner kicks off immediately.
                        Otherwise wait for the next LaunchAgent tick.
    --dry-run         — Report counts only, no enqueue.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))


def _log(msg: str) -> None:
    sys.stderr.write(f"[backfill-orphans] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
    sys.stderr.flush()


def find_orphan_nodes(
    db: sqlite3.Connection, min_mentions: int = 2
) -> list[sqlite3.Row]:
    """Graph nodes that appear in no edge (neither source nor target).

    Filters by `min_mentions` so we don't waste Ollama on one-off mentions.
    """
    return db.execute(
        """SELECT n.id, n.name, n.type, n.mention_count
             FROM graph_nodes n
            WHERE n.status = 'active'
              AND n.mention_count >= ?
              AND n.id NOT IN (SELECT source_id FROM graph_edges)
              AND n.id NOT IN (SELECT target_id FROM graph_edges)
            ORDER BY n.mention_count DESC""",
        (min_mentions,),
    ).fetchall()


def knowledge_ids_linked_to_nodes(
    db: sqlite3.Connection, node_ids: list[str]
) -> list[int]:
    """Find knowledge records that link to any of `node_ids`."""
    if not node_ids:
        return []
    placeholders = ",".join("?" * len(node_ids))
    rows = db.execute(
        f"""SELECT DISTINCT kn.knowledge_id
              FROM knowledge_nodes kn
              JOIN knowledge k ON k.id = kn.knowledge_id
             WHERE kn.node_id IN ({placeholders})
               AND k.status = 'active'
               AND k.superseded_by IS NULL
             ORDER BY kn.knowledge_id DESC""",
        node_ids,
    ).fetchall()
    return [r[0] for r in rows]


def enqueue(db: sqlite3.Connection, knowledge_ids: list[int]) -> int:
    """Enqueue knowledge ids into triple_extraction_queue. Idempotent."""
    from triple_extraction_queue import TripleExtractionQueue

    q = TripleExtractionQueue(db)
    added = 0
    for kid in knowledge_ids:
        if q.enqueue(kid):
            added += 1
    return added


def main() -> int:
    limit = 200
    min_mentions = 2
    trigger_now = False
    dry_run = False

    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif arg.startswith("--min-mentions="):
            min_mentions = int(arg.split("=", 1)[1])
        elif arg == "--trigger-now":
            trigger_now = True
        elif arg == "--dry-run":
            dry_run = True
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0

    memory_dir = Path(os.environ.get("CLAUDE_MEMORY_DIR", Path.home() / ".claude-memory"))
    db_path = memory_dir / "memory.db"
    if not db_path.exists():
        _log(f"db not found at {db_path}")
        return 1

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")

    _log(f"scanning orphans (min_mentions>={min_mentions})")
    orphans = find_orphan_nodes(db, min_mentions=min_mentions)
    total_orphans = len(orphans)
    orphans = orphans[:limit]
    _log(f"found {total_orphans} orphans, processing first {len(orphans)}")

    if not orphans:
        _log("nothing to do")
        return 0

    # Show a few samples for visibility
    for r in orphans[:8]:
        _log(f"  sample: {r['name']} ({r['type']}, mentions={r['mention_count']})")
    if len(orphans) > 8:
        _log(f"  …and {len(orphans) - 8} more")

    node_ids = [r["id"] for r in orphans]
    knowledge_ids = knowledge_ids_linked_to_nodes(db, node_ids)
    _log(f"linked to {len(knowledge_ids)} active knowledge records")

    if dry_run:
        _log("dry-run — not enqueueing")
        db.close()
        return 0

    added = enqueue(db, knowledge_ids)
    _log(f"enqueued {added} new items into triple_extraction_queue")

    if trigger_now:
        trigger_path = memory_dir / ".reflect-pending"
        trigger_path.touch()
        _log(f"touched {trigger_path} — LaunchAgent should drain within ~6s")
    else:
        _log("waiting for next LaunchAgent tick (or run with --trigger-now)")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
