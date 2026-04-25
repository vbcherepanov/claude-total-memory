"""Temporal index for knowledge records (Claude Total Memory v8).

Builds a side table `temporal_index` mapping knowledge_id -> (ts_from, ts_to)
so date-range queries can be answered by pure SQL instead of scanning every
record's content. Parsing is delegated to `temporal_filter`.

Schema is idempotent; indexing is idempotent (INSERT OR REPLACE). The module
is standalone: no edits to server.py required, no global state, thread-safe
as long as each caller passes its own sqlite3 connection.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

from temporal_filter import DateRange, extract_entry_date, parse_query_dates


# --- Config --------------------------------------------------------------- #

_ENV_FLAG = "MEMORY_TEMPORAL_INDEX"
_POINT_WIDTH = timedelta(hours=1)           # width for entry_prefix points
_PROXIMITY_WINDOW_DAYS = 180                # decay window for proximity score


def is_enabled() -> bool:
    """Feature flag: temporal index is opt-in via MEMORY_TEMPORAL_INDEX=1."""
    return os.environ.get(_ENV_FLAG, "0").strip() not in ("", "0", "false", "False")


# --- Schema --------------------------------------------------------------- #

_DDL_TABLE = """
CREATE TABLE IF NOT EXISTS temporal_index (
  knowledge_id INTEGER PRIMARY KEY,
  ts_from      INTEGER NOT NULL,
  ts_to        INTEGER NOT NULL,
  source       TEXT    NOT NULL
)
"""

_DDL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_temporal_range
ON temporal_index(ts_from, ts_to)
"""


def ensure_schema(db: sqlite3.Connection) -> None:
    """Create temporal_index table + range index if absent."""
    db.execute(_DDL_TABLE)
    db.execute(_DDL_INDEX)
    db.commit()


# --- Tag parsing ---------------------------------------------------------- #

_TAG_YEAR_MONTH_RE = re.compile(r"^(\d{4})-(\d{1,2})$")
_TAG_YEAR_RE = re.compile(r"^(\d{4})$")
_TAG_QUARTER_RE = re.compile(r"^[Qq]([1-4])[-_ ]?(\d{4})$")

_QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


def _parse_tag(tag: str) -> Optional[DateRange]:
    """Map tags like '2023-03', '2023', 'Q1-2024' to a DateRange."""
    t = tag.strip()
    if not t:
        return None

    m = _TAG_YEAR_MONTH_RE.match(t)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            start = datetime(y, mo, 1)
            end = (datetime(y + 1, 1, 1) if mo == 12
                   else datetime(y, mo + 1, 1)) - timedelta(seconds=1)
            return DateRange(start, end)

    m = _TAG_YEAR_RE.match(t)
    if m:
        y = int(m.group(1))
        return DateRange(datetime(y, 1, 1),
                         datetime(y, 12, 31, 23, 59, 59))

    m = _TAG_QUARTER_RE.match(t)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        lo, hi = _QUARTER_MONTHS[q]
        start = datetime(y, lo, 1)
        end = (datetime(y + 1, 1, 1) if hi == 12
               else datetime(y, hi + 1, 1)) - timedelta(seconds=1)
        return DateRange(start, end)

    return None


# --- Indexing ------------------------------------------------------------- #

_WARNED_ONCE = False


def _warn_once(msg: str) -> None:
    global _WARNED_ONCE
    if not _WARNED_ONCE:
        print(f"[temporal_index] {msg}", file=sys.stderr)
        _WARNED_ONCE = True


def _widest_range(ranges: list[DateRange]) -> Optional[DateRange]:
    if not ranges:
        return None
    start = min(r.start for r in ranges)
    end = max(r.end for r in ranges)
    return DateRange(start, end)


def _resolve_range(
    content: str, tags: Optional[list[str]]
) -> tuple[Optional[DateRange], str]:
    """Pick best date range for a record. Returns (range, source) or (None, '')."""
    # 1. entry_prefix — most accurate, wins if available
    try:
        dt = extract_entry_date(content or "")
    except Exception:
        dt = None
    if dt is not None:
        return DateRange(dt, dt + _POINT_WIDTH), "entry_prefix"

    # 2. content_regex — union of all mentioned dates
    try:
        found = parse_query_dates(content or "") if content else []
    except Exception:
        found = []
    wide = _widest_range(found)
    if wide is not None:
        return wide, "content_regex"

    # 3. tags — structured hints
    if tags:
        tag_ranges: list[DateRange] = []
        for t in tags:
            try:
                r = _parse_tag(t)
            except Exception:
                r = None
            if r is not None:
                tag_ranges.append(r)
        wide = _widest_range(tag_ranges)
        if wide is not None:
            return wide, "tag"

    return None, ""


def index_record(
    db: sqlite3.Connection,
    knowledge_id: int,
    content: str,
    tags: Optional[list[str]] = None,
) -> bool:
    """Parse dates from a record and upsert into temporal_index. Returns True if stored."""
    try:
        rng, source = _resolve_range(content, tags)
    except Exception as exc:
        _warn_once(f"parse error on id={knowledge_id}: {exc}")
        return False

    if rng is None:
        return False

    ts_from = int(rng.start.timestamp())
    ts_to = int(rng.end.timestamp())
    if ts_to < ts_from:
        ts_to = ts_from

    try:
        db.execute(
            "INSERT OR REPLACE INTO temporal_index "
            "(knowledge_id, ts_from, ts_to, source) VALUES (?, ?, ?, ?)",
            (int(knowledge_id), ts_from, ts_to, source),
        )
        return True
    except Exception as exc:
        _warn_once(f"insert error on id={knowledge_id}: {exc}")
        return False


def bulk_index(
    db: sqlite3.Connection,
    limit: int = 0,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Index every active knowledge record. progress_fn(done, indexed) every 500."""
    started = time.time()
    ensure_schema(db)

    sql = "SELECT id, content, tags FROM knowledge WHERE status='active' ORDER BY id"
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"

    indexed = 0
    skipped = 0
    processed = 0

    for row in db.execute(sql):
        processed += 1
        kid = row[0]
        content = row[1] or ""
        raw_tags = row[2]
        tags: Optional[list[str]] = None
        if raw_tags:
            if isinstance(raw_tags, str):
                # Try JSON, fall back to comma split
                try:
                    import json
                    parsed = json.loads(raw_tags)
                    if isinstance(parsed, list):
                        tags = [str(x) for x in parsed]
                    else:
                        tags = [raw_tags]
                except Exception:
                    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
            elif isinstance(raw_tags, (list, tuple)):
                tags = [str(x) for x in raw_tags]

        ok = index_record(db, kid, content, tags)
        if ok:
            indexed += 1
        else:
            skipped += 1

        if progress_fn and processed % 500 == 0:
            try:
                progress_fn(processed, indexed)
            except Exception:
                pass

    db.commit()
    return {
        "indexed": indexed,
        "skipped": skipped,
        "elapsed_sec": round(time.time() - started, 3),
    }


# --- Query ---------------------------------------------------------------- #

def filter_by_query_date(
    db: sqlite3.Connection, query: str
) -> Optional[set[int]]:
    """Return knowledge IDs whose indexed range overlaps any query date range.

    Returns None when the query has no parseable dates — caller should treat
    that as "no temporal filter" and skip it.
    """
    try:
        ranges = parse_query_dates(query or "")
    except Exception:
        ranges = []
    if not ranges:
        return None

    result: set[int] = set()
    for r in ranges:
        q_start = int(r.start.timestamp())
        q_end = int(r.end.timestamp())
        # Overlap: ts_from <= q_end AND ts_to >= q_start
        cur = db.execute(
            "SELECT knowledge_id FROM temporal_index "
            "WHERE ts_from <= ? AND ts_to >= ?",
            (q_end, q_start),
        )
        for (kid,) in cur:
            result.add(int(kid))
    return result


def date_proximity_score(
    db: sqlite3.Connection, knowledge_id: int, query_date: datetime
) -> float:
    """1.0 = query_date falls inside the record's range; decays linearly over 180d."""
    cur = db.execute(
        "SELECT ts_from, ts_to FROM temporal_index WHERE knowledge_id = ?",
        (int(knowledge_id),),
    )
    row = cur.fetchone()
    if not row:
        return 0.0
    ts_from, ts_to = int(row[0]), int(row[1])
    q = int(query_date.timestamp())
    if ts_from <= q <= ts_to:
        return 1.0
    dist_sec = (ts_from - q) if q < ts_from else (q - ts_to)
    dist_days = dist_sec / 86400.0
    window = float(_PROXIMITY_WINDOW_DAYS)
    if dist_days >= window:
        return 0.0
    return max(0.0, 1.0 - dist_days / window)


# --- Smoke test ----------------------------------------------------------- #

def _smoke() -> None:
    import tempfile
    import json

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = sqlite3.connect(tmp.name)
    db.execute(
        "CREATE TABLE knowledge ("
        " id INTEGER PRIMARY KEY, content TEXT, tags TEXT,"
        " status TEXT DEFAULT 'active', created_at INTEGER)"
    )
    fixtures = [
        (1, "[1:56 pm on 8 May, 2023] Caroline: Hey Mel!",
         json.dumps(["chat"])),
        (2, "[2023-03-15 09:00] Bob: we deployed the new API today",
         json.dumps(["deploy"])),
        (3, "Meeting notes: in March 2023 we started the project. "
            "Follow-up scheduled for April 2023.",
         json.dumps(["meeting"])),
        (4, "Random note without any dates mentioned at all.",
         json.dumps(["misc"])),
        (5, "Retrospective document.",
         json.dumps(["2023-03", "retro"])),
    ]
    db.executemany(
        "INSERT INTO knowledge (id, content, tags) VALUES (?,?,?)", fixtures
    )
    db.commit()

    ensure_schema(db)
    stats = bulk_index(db)
    print("bulk_index stats:", stats)

    rows = list(db.execute(
        "SELECT knowledge_id, ts_from, ts_to, source FROM temporal_index "
        "ORDER BY knowledge_id"
    ))
    print("index rows:")
    for r in rows:
        kid, tf, tt, src = r
        print(f"  id={kid} from={datetime.fromtimestamp(tf).isoformat()} "
              f"to={datetime.fromtimestamp(tt).isoformat()} src={src}")

    ids = filter_by_query_date(db, "what happened in March 2023")
    print("matching IDs for 'what happened in March 2023':",
          sorted(ids) if ids is not None else None)

    score = date_proximity_score(db, 2, datetime(2023, 3, 15))
    print(f"proximity(id=2, 2023-03-15) = {score:.3f}")

    none_case = filter_by_query_date(db, "tell me about cats")
    print("no-date query returns:", none_case)

    db.close()
    os.unlink(tmp.name)


if __name__ == "__main__":
    _smoke()
