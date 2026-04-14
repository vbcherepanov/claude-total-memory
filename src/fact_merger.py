"""Semantic fact merger — consolidate related-but-distinct facts via LLM.

Complements `reflection.digest.merge_duplicates` (which handles near-duplicates
at Jaccard >=0.85). This module finds clusters of related records — cosine
similarity in the 0.70-0.95 band — and asks an LLM to synthesize them into a
single consolidated fact. Validator guards against LLM information loss.

Example:
    "User uses Go for backend" + "User builds APIs in Go"
    → "User's primary backend language is Go (used for APIs)."

Source rows are archived (status='archived', superseded_by=<merged_id>) and
the merge event recorded in `knowledge_merges` for audit/rollback.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Callable

try:
    from validator import ContentValidator
except ImportError:  # when imported as package
    from .validator import ContentValidator  # type: ignore[no-redef]

LOG = lambda msg: sys.stderr.write(f"[fact-merger] {msg}\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SimilarityFn = Callable[[int, int], float]
LLMMergeFn = Callable[[list[str]], str]


class FactMerger:
    """Find and merge semantically related facts via LLM consolidation."""

    def __init__(
        self,
        db: sqlite3.Connection,
        similarity_fn: SimilarityFn,
        llm_merge_fn: LLMMergeFn | None = None,
    ) -> None:
        """
        Args:
            db: SQLite connection (row_factory = Row).
            similarity_fn: (id_a, id_b) -> cosine similarity in [0, 1].
                Typically wraps server._binary_search / float32 cosine.
            llm_merge_fn: (list[content]) -> merged_content. If None, no merges
                happen (useful for tests that only want clustering).
        """
        self.db = db
        self.similarity = similarity_fn
        self.llm_merge = llm_merge_fn or (lambda _c: "")
        self.validator = ContentValidator()

    # ──────────────────────────────────────────────
    # Cluster discovery
    # ──────────────────────────────────────────────

    def find_clusters(
        self,
        project: str | None = None,
        min_similarity: float = 0.72,
        max_similarity: float = 0.95,
        max_cluster_size: int = 5,
    ) -> list[list[int]]:
        """Find clusters of related (but not duplicate) knowledge records.

        Simple agglomerative: for each candidate pair with similarity in
        [min, max], union-find into a cluster. Caps each cluster at
        `max_cluster_size`.
        """
        rows = self._candidate_rows(project)
        ids = [r["id"] for r in rows]

        # Union-Find for grouping
        parent: dict[int, int] = {i: i for i in ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                try:
                    sim = float(self.similarity(a, b))
                except Exception as e:  # noqa: BLE001
                    LOG(f"similarity({a},{b}) failed: {e}")
                    continue
                if min_similarity <= sim <= max_similarity:
                    union(a, b)

        # Collect clusters
        groups: dict[int, list[int]] = {}
        for i in ids:
            root = find(i)
            groups.setdefault(root, []).append(i)

        clusters = [sorted(g) for g in groups.values() if len(g) > 1]

        # Respect max_cluster_size by splitting oversized groups
        capped: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= max_cluster_size:
                capped.append(cl)
            else:
                for start in range(0, len(cl), max_cluster_size):
                    chunk = cl[start : start + max_cluster_size]
                    if len(chunk) > 1:
                        capped.append(chunk)

        return capped

    def _candidate_rows(self, project: str | None) -> list[sqlite3.Row]:
        if project:
            return self.db.execute(
                "SELECT id, content FROM knowledge "
                "WHERE status='active' AND project=? AND superseded_by IS NULL "
                "ORDER BY id",
                (project,),
            ).fetchall()
        return self.db.execute(
            "SELECT id, content FROM knowledge "
            "WHERE status='active' AND superseded_by IS NULL ORDER BY id"
        ).fetchall()

    # ──────────────────────────────────────────────
    # Merge a single cluster
    # ──────────────────────────────────────────────

    def merge_cluster(self, ids: list[int]) -> dict[str, Any]:
        """Synthesize a consolidated knowledge record from a cluster.

        Returns {"merged_id": int|None, "reason": str}. If the LLM output
        fails validation against the concatenated source content (loses URLs,
        paths, inline code), abort: sources stay active, no merged record.
        """
        if len(ids) < 2:
            return {"merged_id": None, "reason": "cluster too small"}

        rows = self._fetch_rows(ids)
        if len(rows) < 2:
            return {"merged_id": None, "reason": "sources not found"}

        contents = [r["content"] for r in rows]

        try:
            merged_text = self.llm_merge(contents)
        except Exception as e:  # noqa: BLE001
            LOG(f"llm_merge failed: {e}")
            return {"merged_id": None, "reason": f"llm error: {e}"}

        if not merged_text or not merged_text.strip():
            return {"merged_id": None, "reason": "llm returned empty"}

        # Validate: merged must preserve critical elements from combined source
        combined = "\n\n".join(contents)
        v = self.validator.validate(combined, merged_text)
        if not v.ok:
            LOG(f"validator rejected merge: {v.errors}")
            return {
                "merged_id": None,
                "reason": f"validator rejected: {'; '.join(v.errors[:3])}",
            }

        # Insert merged record
        first = rows[0]
        merged_id = self.db.execute(
            """INSERT INTO knowledge
                 (content, project, type, tags, status, confidence, created_at, updated_at)
               VALUES (?, ?, 'fact', ?, 'active', ?, ?, ?)""",
            (
                merged_text.strip(),
                first["project"] if "project" in first.keys() else "general",
                json.dumps(["merged", "consolidated"]),
                max((r["confidence"] for r in rows if r["confidence"] is not None), default=1.0),
                _now(),
                _now(),
            ),
        ).lastrowid

        # Archive sources
        for r in rows:
            self.db.execute(
                "UPDATE knowledge SET status='archived', superseded_by=?, updated_at=? "
                "WHERE id=?",
                (merged_id, _now(), r["id"]),
            )

        # Audit trail
        self.db.execute(
            "INSERT INTO knowledge_merges (merged_knowledge_id, source_ids, rationale, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                merged_id,
                json.dumps([r["id"] for r in rows]),
                "semantic fact merge (cosine 0.72-0.95)",
                _now(),
            ),
        )
        self.db.commit()

        LOG(f"merged cluster {ids} -> knowledge_id={merged_id}")
        return {"merged_id": merged_id, "reason": "ok"}

    def _fetch_rows(self, ids: list[int]) -> list[sqlite3.Row]:
        placeholders = ",".join("?" * len(ids))
        return self.db.execute(
            f"SELECT id, content, project, confidence FROM knowledge "
            f"WHERE id IN ({placeholders}) AND status='active'",
            ids,
        ).fetchall()

    # ──────────────────────────────────────────────
    # Run (drive the full loop)
    # ──────────────────────────────────────────────

    def run(self, project: str | None = None) -> dict[str, int]:
        stats = {"clusters_found": 0, "merged": 0, "rejected": 0}

        clusters = self.find_clusters(project=project)
        stats["clusters_found"] = len(clusters)

        for cluster in clusters:
            result = self.merge_cluster(cluster)
            if result["merged_id"] is not None:
                stats["merged"] += 1
            else:
                stats["rejected"] += 1

        LOG(f"fact_merger.run: {stats}")
        return stats
