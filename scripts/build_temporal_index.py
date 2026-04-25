#!/usr/bin/env python3
"""Extract dates from synthesized_fact rows and build a temporal lookup.

LoCoMo temporal questions ("When did X happen?") need exact date recall.
Semantic search + temporal_rerank currently gets 0.302 Acc on temporal
category. A structured (event_entity, date) table we look up first should
lift that materially.

Schema:
    fact_temporal(
        knowledge_id INTEGER,
        project TEXT,
        entity TEXT,          -- lowercase main entity (person/place)
        event_hint TEXT,      -- lowercased verb phrase, best-effort
        date_iso TEXT,        -- ISO "YYYY-MM-DD" if resolved, else ""
        date_raw TEXT         -- original surface form ("May 2023", "last fall")
    )

Fast regex + a few heuristics — no LLM call. Runs in a couple seconds.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path


MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "sept": "09", "oct": "10",
    "nov": "11", "dec": "12",
}

DATE_PATTERNS = [
    # 2023-03-15, 2023-3-15
    (re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b"),
     lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
    # 15 March 2023, 15th March 2023
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
                r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(20\d{2})\b",
                re.I),
     lambda m: f"{m.group(3)}-{MONTHS[m.group(2).lower()]}-{int(m.group(1)):02d}"),
    # March 15, 2023 / March 15 2023
    (re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b",
                re.I),
     lambda m: f"{m.group(3)}-{MONTHS[m.group(1).lower()]}-{int(m.group(2)):02d}"),
    # March 2023
    (re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
                re.I),
     lambda m: f"{m.group(2)}-{MONTHS[m.group(1).lower()]}-01"),
    # 2023 (year only — last resort)
    (re.compile(r"\b(20\d{2})\b"),
     lambda m: f"{m.group(1)}-01-01"),
]

STOP_ENTITIES = frozenset({
    "the", "a", "an", "i", "you", "he", "she", "they", "we",
})


def extract_date(text: str) -> tuple[str, str] | None:
    """Return (iso, raw) of the first high-confidence date, else None."""
    for rx, mk_iso in DATE_PATTERNS:
        m = rx.search(text)
        if m:
            try:
                iso = mk_iso(m)
                return iso, m.group(0)
            except (KeyError, ValueError):
                continue
    return None


ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*)\b")


def extract_entity(text: str) -> str:
    """Pull the first capitalised name — best-effort subject of the fact."""
    for m in ENTITY_RE.finditer(text):
        chunk = m.group(1).strip()
        if chunk.lower() not in STOP_ENTITIES and len(chunk) > 1:
            return chunk
    return ""


def extract_event_hint(text: str, entity: str) -> str:
    """Short verb phrase following the subject, lowercased. Fallback empty."""
    if not entity:
        return ""
    lower_text = text.lower()
    lower_ent = entity.lower()
    idx = lower_text.find(lower_ent)
    if idx == -1:
        return ""
    tail = text[idx + len(entity):].strip().lower()
    # Drop connectors.
    tail = re.sub(r"^(is|was|will be|did|has|had|have|went|goes)\s+", "", tail)
    # Keep first ~6 words.
    words = tail.split()[:6]
    return " ".join(words).rstrip(".,;:!?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="/tmp/locomo_bench_db")
    ap.add_argument("--project-prefix", default="locomo_")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    db_file = f"{args.db_path}/memory.db"
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row

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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ft_date    ON fact_temporal(date_iso)")

    if args.reset:
        n = conn.execute("DELETE FROM fact_temporal WHERE project LIKE ?",
                         (f"{args.project_prefix}%",)).rowcount
        conn.commit()
        print(f"[temporal] reset: {n} rows deleted")

    rows = conn.execute(
        "SELECT id, project, content FROM knowledge "
        "WHERE type='synthesized_fact' AND status='active' AND project LIKE ?",
        (f"{args.project_prefix}%",),
    ).fetchall()
    print(f"[temporal] scanning {len(rows)} synth_facts")

    inserted = 0
    for r in rows:
        content = (r["content"] or "").strip()
        if not content:
            continue
        date = extract_date(content)
        if not date:
            continue
        iso, raw = date
        entity = extract_entity(content)
        hint = extract_event_hint(content, entity)
        conn.execute(
            "INSERT OR REPLACE INTO fact_temporal "
            "(knowledge_id, project, entity, event_hint, date_iso, date_raw) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (int(r["id"]), r["project"], entity.lower(), hint, iso, raw),
        )
        inserted += 1
    conn.commit()

    print(f"[temporal] indexed {inserted} dated facts")
    # Quick stats
    stats = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT project), COUNT(DISTINCT entity) "
        "FROM fact_temporal"
    ).fetchone()
    print(f"[temporal] total={stats[0]} across {stats[1]} projects, {stats[2]} entities")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
