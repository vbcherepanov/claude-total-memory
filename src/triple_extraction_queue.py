"""Async triple extraction queue.

`memory_save` enqueues `knowledge_id` here (<1ms). A background worker pulls
pending items and runs `ConceptExtractor.extract_and_link(deep=True)` which
calls Ollama and persists (subject, predicate, object) triples into
`graph_edges`. Separates the hot save path from the slow LLM extraction.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Callable

LOG = lambda msg: sys.stderr.write(f"[triple-queue] {msg}\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Extractor callable signature: (knowledge_id, content) -> extraction_result_dict
# Implementations typically wrap ConceptExtractor.extract_and_link(deep=True).
ExtractorFn = Callable[[int, str], dict[str, Any]]


class TripleExtractionQueue:
    """Durable FIFO queue for deep (LLM) triple extraction."""

    def __init__(
        self,
        db: sqlite3.Connection,
        max_attempts: int = 3,
    ) -> None:
        self.db = db
        self.max_attempts = max(1, max_attempts)

    # ──────────────────────────────────────────────
    # Enqueue
    # ──────────────────────────────────────────────

    def enqueue(self, knowledge_id: int) -> bool:
        """Add `knowledge_id` to the pending queue.

        Idempotent: if a pending row already exists for this knowledge_id, no
        new row is created. A done/failed row does NOT block a new enqueue
        (e.g. content update should retrigger extraction).

        Returns True if a new row was inserted, False if skipped.
        """
        existing = self.db.execute(
            "SELECT id FROM triple_extraction_queue "
            "WHERE knowledge_id = ? AND status = 'pending' LIMIT 1",
            (knowledge_id,),
        ).fetchone()
        if existing:
            return False
        self.db.execute(
            "INSERT INTO triple_extraction_queue (knowledge_id, status, created_at) "
            "VALUES (?, 'pending', ?)",
            (knowledge_id, _now()),
        )
        self.db.commit()
        return True

    # ──────────────────────────────────────────────
    # Claim / finalize
    # ──────────────────────────────────────────────

    def claim_next(self) -> dict | None:
        """Atomically mark the oldest pending row as processing and return it.

        Returns dict with id, knowledge_id, attempts. None if no pending work.
        """
        # SQLite lacks native SKIP LOCKED, but this process is single-threaded
        # per queue instance; reflection.agent runs sequentially.
        row = self.db.execute(
            "SELECT id, knowledge_id, attempts FROM triple_extraction_queue "
            "WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        self.db.execute(
            "UPDATE triple_extraction_queue "
            "SET status = 'processing', claimed_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        self.db.commit()
        return {
            "id": row["id"],
            "knowledge_id": row["knowledge_id"],
            "attempts": row["attempts"],
            "status": "processing",
        }

    def mark_done(self, item_id: int) -> None:
        # Remove any stale 'done' row for the same knowledge_id to avoid
        # UNIQUE(knowledge_id, status) conflict when backfill re-enqueues.
        row = self.db.execute(
            "SELECT knowledge_id FROM triple_extraction_queue WHERE id=?",
            (item_id,),
        ).fetchone()
        if row is not None:
            self.db.execute(
                "DELETE FROM triple_extraction_queue "
                "WHERE knowledge_id = ? AND status = 'done' AND id != ?",
                (row["knowledge_id"], item_id),
            )
        self.db.execute(
            "UPDATE triple_extraction_queue "
            "SET status = 'done', processed_at = ? WHERE id = ?",
            (_now(), item_id),
        )
        self.db.commit()

    def mark_failed(self, item_id: int, error: str) -> None:
        """Bump attempts. If we've hit max_attempts, mark failed; otherwise
        requeue as pending so another worker picks it up."""
        row = self.db.execute(
            "SELECT attempts, knowledge_id FROM triple_extraction_queue WHERE id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            return
        next_attempts = (row["attempts"] or 0) + 1
        new_status = "failed" if next_attempts >= self.max_attempts else "pending"
        # Remove any stale row with the same (knowledge_id, target_status) to
        # avoid UNIQUE conflict when re-enqueued content cycles through states.
        self.db.execute(
            "DELETE FROM triple_extraction_queue "
            "WHERE knowledge_id = ? AND status = ? AND id != ?",
            (row["knowledge_id"], new_status, item_id),
        )
        self.db.execute(
            "UPDATE triple_extraction_queue "
            "SET status = ?, attempts = ?, last_error = ?, processed_at = ? "
            "WHERE id = ?",
            (new_status, next_attempts, (error or "")[:500], _now(), item_id),
        )
        self.db.commit()

    # ──────────────────────────────────────────────
    # Worker loop
    # ──────────────────────────────────────────────

    def process_pending(
        self,
        extractor: ExtractorFn,
        limit: int = 10,
    ) -> dict[str, int]:
        """Drain up to `limit` pending items, calling `extractor(kid, content)`
        for each. Returns stats {processed, failed, skipped}."""
        stats = {"processed": 0, "failed": 0, "skipped": 0}

        for _ in range(limit):
            item = self.claim_next()
            if item is None:
                break

            kid = item["knowledge_id"]
            content_row = self.db.execute(
                "SELECT content FROM knowledge WHERE id = ?", (kid,)
            ).fetchone()
            if content_row is None:
                # Knowledge gone — drop silently
                self.mark_done(item["id"])
                stats["skipped"] += 1
                continue

            try:
                extractor(kid, content_row["content"] or "")
                self.mark_done(item["id"])
                stats["processed"] += 1
            except Exception as e:  # noqa: BLE001 — we deliberately want to retry
                LOG(f"extract failed for knowledge_id={kid}: {e}")
                self.mark_failed(item["id"], str(e))
                stats["failed"] += 1

        return stats

    # ──────────────────────────────────────────────
    # Observability
    # ──────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        rows = self.db.execute(
            "SELECT status, COUNT(*) AS c FROM triple_extraction_queue GROUP BY status"
        ).fetchall()
        out = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
        for r in rows:
            out[r["status"]] = r["c"]
        return out
