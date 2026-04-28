"""v11.0 — handlers for new MCP tools.

Thin wrappers that register cleanly in server.py without bloating its
already 6000+ line monolith. Each handler returns a JSON-serialisable
dict ready for `mcp.types.TextContent` packaging.

Tools added:
  - memory_recall_iterative      (W1-B IRCoT)
  - memory_temporal_query        (W1-C Allen + arithmetic)
  - memory_entity_resolve        (W1-F entity resolver)
  - memory_consolidate_status    (W2-G consolidation daemon)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import sys

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ─── memory_recall_iterative (IRCoT) ────────────────────────────────────


def handle_recall_iterative(
    args: dict,
    *,
    search_fn: Callable,
) -> dict:
    """IRCoT-style iterative retrieval over the standard recall pipeline."""
    from ai_layer.iterative_retriever import iterative_retrieve

    query = args.get("query") or ""
    if not query.strip():
        return {"error": "query is required"}

    project = args.get("project")
    max_iters = int(args.get("max_iters") or 4)
    k_per_iter = int(args.get("k_per_iter") or 5)
    llm_model = args.get("llm_model") or "haiku"

    def _adapter(q: str, k: int = 10, project: str | None = None) -> list[dict]:
        result = search_fn(q, project, "all", k)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        return []

    res = iterative_retrieve(
        query,
        search_fn=_adapter,
        project=project,
        max_iters=max_iters,
        k_per_iter=k_per_iter,
        llm_model=llm_model,
    )

    return {
        "query": query,
        "iterations_used": res.iterations_used,
        "terminated_reason": res.terminated_reason,
        "sub_queries": list(res.sub_queries),
        "partial_answers": list(res.partial_answers),
        "evidence_count": len(res.final_evidence),
        "evidence": res.final_evidence[:20],
        "provenance": res.provenance,
    }


# ─── memory_temporal_query (Allen + arithmetic) ─────────────────────────


def handle_temporal_query(args: dict) -> dict:
    """Compute temporal relations and durations."""
    from datetime import datetime as _dt

    op = (args.get("op") or "").strip().lower()
    if not op:
        return {"error": "op is required (one of: relation, duration_between, normalize)"}

    if op == "relation":
        from memory_core.temporal.allen import Interval, relation
        a_start = _dt.fromisoformat(args["a_start"])
        a_end = _dt.fromisoformat(args["a_end"])
        b_start = _dt.fromisoformat(args["b_start"])
        b_end = _dt.fromisoformat(args["b_end"])
        ia = Interval(a_start, a_end)
        ib = Interval(b_start, b_end)
        rel = relation(ia, ib)
        return {"op": "relation", "relation": rel.value if hasattr(rel, "value") else str(rel)}

    if op == "duration_between":
        from memory_core.temporal.arithmetic import (
            duration_between,
            days_between,
            weeks_between,
            months_between,
            years_between,
            format_human,
        )
        a = _dt.fromisoformat(args["a"])
        b = _dt.fromisoformat(args["b"])
        td = duration_between(a, b)
        return {
            "op": "duration_between",
            "seconds": td.total_seconds(),
            "days": days_between(a, b),
            "weeks": weeks_between(a, b),
            "months": months_between(a, b),
            "years": years_between(a, b),
            "human_en": format_human(td, lang="en"),
            "human_ru": format_human(td, lang="ru"),
        }

    if op == "normalize":
        from memory_core.temporal.normalizer import normalize
        phrase = args.get("phrase") or ""
        anchor = _dt.fromisoformat(args.get("anchor") or _dt.now(timezone.utc).isoformat())
        lang = args.get("lang") or "auto"
        nd = normalize(phrase, anchor, lang=lang)
        if nd is None:
            return {"op": "normalize", "phrase": phrase, "result": None}
        return {
            "op": "normalize",
            "phrase": phrase,
            "iso": nd.iso,
            "kind": nd.kind,
            "confidence": nd.confidence,
        }

    return {"error": f"unknown op: {op}"}


# ─── memory_entity_resolve ──────────────────────────────────────────────


def handle_entity_resolve(args: dict, *, conn: sqlite3.Connection, embed_fn: Callable) -> dict:
    """Resolve a mention to a canonical entity within a project+type."""
    from memory_core.entity_resolver import resolve

    mention = (args.get("mention") or "").strip()
    project = args.get("project") or "general"
    type_ = args.get("type") or "person"
    threshold = float(args.get("threshold") or 0.85)
    create = bool(args.get("create_if_missing", True))

    if not mention:
        return {"error": "mention is required"}

    def _embed(text: str):
        try:
            import numpy as np
            v = embed_fn(text)
            if v is None:
                return np.zeros(0, dtype="float32")
            return np.asarray(v, dtype="float32")
        except Exception:
            import numpy as np
            return np.zeros(0, dtype="float32")

    res = resolve(
        conn, mention, project, type_,
        embed_fn=_embed,
        threshold=threshold,
        create_if_missing=create,
    )
    return {
        "canonical_id": res.canonical_id,
        "canonical_name": res.canonical_name,
        "matched_via": res.matched_via,
        "confidence": res.confidence,
        "is_new": res.is_new,
    }


# ─── memory_consolidate_status ──────────────────────────────────────────


def handle_consolidate_status(args: dict, *, conn: sqlite3.Connection) -> dict:
    """Return the current consolidation daemon state across projects."""
    rows = conn.execute(
        "SELECT project, last_consolidated_at, last_status, last_error, locked_until, stats_json, updated_at "
        "FROM consolidation_state ORDER BY updated_at DESC"
    ).fetchall()

    projects = []
    now = datetime.now(timezone.utc)
    for row in rows:
        d = dict(row)
        locked = d.get("locked_until")
        active_lock = False
        if locked:
            try:
                lt = datetime.fromisoformat(locked.replace("Z", "+00:00"))
                if lt.tzinfo is None:
                    lt = lt.replace(tzinfo=timezone.utc)
                active_lock = lt > now
            except (ValueError, TypeError):
                pass
        stats = None
        sj = d.get("stats_json")
        if sj:
            try:
                stats = json.loads(sj)
            except (json.JSONDecodeError, TypeError):
                stats = {"raw": sj}
        projects.append({
            "project": d.get("project"),
            "last_consolidated_at": d.get("last_consolidated_at"),
            "last_status": d.get("last_status"),
            "last_error": d.get("last_error"),
            "active_lock": active_lock,
            "locked_until": locked,
            "stats": stats,
            "updated_at": d.get("updated_at"),
        })

    activity_rows = conn.execute(
        "SELECT project, last_touched_at, touch_count_24h FROM project_activity "
        "ORDER BY last_touched_at DESC LIMIT 20"
    ).fetchall()
    activity = [dict(r) for r in activity_rows]

    return {
        "projects": projects,
        "recent_activity": activity,
        "as_of": now.isoformat(),
    }
