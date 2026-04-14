"""Post-search filter using deep_enrichment metadata.

Apply an AND-filter over a list of candidate knowledge_ids based on the
entities/intent/topics persisted by deep_enrichment_queue into
`knowledge_enrichment`. Unfiltered records (no enrichment row) are excluded
whenever any filter is active — enforces a "must have enrichment to qualify"
contract that avoids false positives.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from typing import Iterable

LOG = lambda msg: sys.stderr.write(f"[enrich-filter] {msg}\n")


def filter_by_enrichment(
    db: sqlite3.Connection,
    candidate_ids: list[int],
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    intent: str | None = None,
) -> list[int]:
    """Keep only candidates whose enrichment matches ALL given filters.

    Matching rules:
      - topics: OR within the list ("matches any topic")
      - entities: OR within the list (name-only, case-insensitive)
      - intent: exact string match
      - topics + entities + intent: AND between categories

    Input order is preserved in the output.
    """
    if not candidate_ids:
        return []

    any_filter = bool(topics) or bool(entities) or intent
    if not any_filter:
        return list(candidate_ids)

    wanted_topics = {t.lower() for t in (topics or []) if t}
    wanted_entities = {e.lower() for e in (entities or []) if e}

    placeholders = ",".join("?" * len(candidate_ids))
    try:
        rows = db.execute(
            f"""SELECT knowledge_id, entities, intent, topics
                  FROM knowledge_enrichment
                 WHERE knowledge_id IN ({placeholders})""",
            list(candidate_ids),
        ).fetchall()
    except sqlite3.Error as e:
        LOG(f"query error: {e}")
        return list(candidate_ids)  # fail open: don't lose all results

    enrichment_by_id: dict[int, dict] = {}
    for r in rows:
        try:
            enrichment_by_id[r["knowledge_id"]] = {
                "entities": json.loads(r["entities"] or "[]"),
                "intent": r["intent"] or "unknown",
                "topics": json.loads(r["topics"] or "[]"),
            }
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    kept: list[int] = []
    for kid in candidate_ids:
        enr = enrichment_by_id.get(kid)
        if enr is None:
            # No enrichment row — excluded when any filter is active
            continue
        if not _matches(enr, wanted_topics, wanted_entities, intent):
            continue
        kept.append(kid)
    return kept


def _matches(
    enr: dict,
    wanted_topics: set[str],
    wanted_entities: set[str],
    intent: str | None,
) -> bool:
    if intent and enr.get("intent") != intent:
        return False

    if wanted_topics:
        have = {str(t).lower() for t in (enr.get("topics") or []) if t}
        if have.isdisjoint(wanted_topics):
            return False

    if wanted_entities:
        have_names: set[str] = set()
        for e in enr.get("entities") or []:
            if isinstance(e, dict):
                name = e.get("name")
                if name:
                    have_names.add(str(name).lower())
            elif isinstance(e, str):
                have_names.add(e.lower())
        if have_names.isdisjoint(wanted_entities):
            return False

    return True
