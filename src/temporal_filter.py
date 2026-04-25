"""Lightweight temporal filter / re-ranker for LoCoMo-style retrieval.

Given a user query and a list of retrieved entries whose content starts
with `[<date-string>] Speaker: ...`, this module:
  1. Parses any date/time cues from the query
  2. Extracts the session date from each entry's content
  3. Re-ranks entries by blending the retrieval score with a
     temporal-proximity score
  4. Optionally drops entries outside the query's date window

Uses deterministic regex + dateutil (no LLM calls, no cost).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

try:
    from dateutil import parser as dateparser
except Exception:
    dateparser = None


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Compact regex patterns (ordered — first match wins)
DATE_PATTERNS = [
    # "26 March 2023", "March 26 2023", "March 26, 2023"
    (r"\b(\d{1,2})\s+(%s)(?:\s*,)?\s+(\d{4})\b" % "|".join(MONTHS), "dmy"),
    (r"\b(%s)\s+(\d{1,2})(?:\s*,)?\s+(\d{4})\b" % "|".join(MONTHS), "mdy"),
    # "in March 2023", "March 2023"
    (r"\b(%s)\s+(\d{4})\b" % "|".join(MONTHS), "my"),
    # ISO date "2023-03-26"
    (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", "iso"),
    # "Q1 2023", "spring 2023"
    (r"\b(spring|summer|fall|autumn|winter)\s+(\d{4})\b", "season"),
    # "in 2023"
    (r"\b(?:in|during)\s+(\d{4})\b", "year"),
]

SEASON_MONTHS = {
    "spring": (3, 5), "summer": (6, 8), "fall": (9, 11),
    "autumn": (9, 11), "winter": (12, 2),
}

TEMPORAL_KEYWORDS = {
    "when", "first", "last", "before", "after", "since", "until", "during",
    "ago", "recently", "earlier", "later", "yesterday", "tomorrow",
    "today", "week", "weekend", "month", "year",
    "когда", "первый", "последний", "до", "после", "с", "недел", "месяц",
    "год", "раньше", "позже",
}


@dataclass
class DateRange:
    start: datetime
    end: datetime

    def contains(self, d: datetime) -> bool:
        return self.start <= d <= self.end

    def distance_days(self, d: datetime) -> float:
        if self.contains(d):
            return 0.0
        if d < self.start:
            return (self.start - d).total_seconds() / 86400
        return (d - self.end).total_seconds() / 86400


def _season_range(season: str, year: int) -> DateRange:
    lo, hi = SEASON_MONTHS[season]
    if lo <= hi:
        return DateRange(datetime(year, lo, 1), datetime(year, hi, 28))
    # winter: Dec of year → Feb of year+1
    return DateRange(datetime(year, lo, 1), datetime(year + 1, hi, 28))


def parse_query_dates(query: str) -> list[DateRange]:
    q = query.lower()
    ranges: list[DateRange] = []
    for pat, kind in DATE_PATTERNS:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            try:
                g = [x.lower() for x in m.groups() if x is not None]
                if kind == "dmy":
                    day, mon, year = int(g[0]), MONTHS[g[1]], int(g[2])
                    d = datetime(year, mon, day)
                    ranges.append(DateRange(d, d + timedelta(days=1)))
                elif kind == "mdy":
                    mon, day, year = MONTHS[g[0]], int(g[1]), int(g[2])
                    d = datetime(year, mon, day)
                    ranges.append(DateRange(d, d + timedelta(days=1)))
                elif kind == "my":
                    mon, year = MONTHS[g[0]], int(g[1])
                    s = datetime(year, mon, 1)
                    # end of month
                    e = (datetime(year + (mon // 12), (mon % 12) + 1, 1)
                         if mon < 12 else datetime(year + 1, 1, 1))
                    ranges.append(DateRange(s, e - timedelta(seconds=1)))
                elif kind == "iso":
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                    dt = datetime(y, mo, d)
                    ranges.append(DateRange(dt, dt + timedelta(days=1)))
                elif kind == "season":
                    ranges.append(_season_range(g[0], int(g[1])))
                elif kind == "year":
                    y = int(g[0])
                    ranges.append(DateRange(datetime(y, 1, 1), datetime(y, 12, 31)))
            except Exception:
                continue
    return ranges


def has_temporal_intent(query: str) -> bool:
    q = query.lower()
    if any(kw in q for kw in TEMPORAL_KEYWORDS):
        return True
    # Month-name mention alone doesn't imply temporal intent, but combined
    # with a year it does — handled via parse_query_dates presence.
    if parse_query_dates(query):
        return True
    return False


_ENTRY_DATE_RE = re.compile(r"^\[([^\]]+)\]")


def extract_entry_date(content: str) -> Optional[datetime]:
    """Parse `[1:56 pm on 8 May, 2023] Speaker: ...` → datetime."""
    if not dateparser:
        return None
    m = _ENTRY_DATE_RE.match(content or "")
    if not m:
        return None
    raw = m.group(1)
    # Normalize "1:56 pm on 8 May, 2023" → "1:56 pm 8 May 2023"
    raw = raw.replace(" on ", " ").replace(",", "")
    try:
        return dateparser.parse(raw, fuzzy=True)
    except Exception:
        return None


def temporal_rerank(
    query: str,
    entries: list[dict],
    *,
    score_weight: float = 0.5,
    proximity_weight: float = 0.5,
    max_window_days: int = 180,
    drop_outside_window: bool = False,
) -> list[dict]:
    """Return entries re-ranked by blending score + date proximity.

    - If no temporal intent in query: returns entries unchanged.
    - Entries with un-parseable dates keep their original score (no bonus,
      no penalty).
    - `drop_outside_window=True` hard-drops entries whose date is further
      than `max_window_days` from any query range.
    """
    if not has_temporal_intent(query):
        return entries

    ranges = parse_query_dates(query)
    if not ranges:
        # Temporal intent without explicit dates — don't rerank
        return entries

    rescored = []
    for e in entries:
        base_score = e.get("score", 0.0)
        date = extract_entry_date(e.get("content", ""))
        if date is None:
            # Unknown date → keep neutral
            rescored.append((e, base_score))
            continue
        # Min distance to any range
        dist = min(r.distance_days(date) for r in ranges)
        if drop_outside_window and dist > max_window_days:
            continue
        # Proximity bonus: 1.0 at distance 0, decays over max_window_days
        proximity = max(0.0, 1.0 - dist / max_window_days)
        # Normalize base_score (our scores can exceed 1.0; soft-clamp at 5)
        norm_base = min(base_score / 5.0, 1.0)
        blended = score_weight * norm_base + proximity_weight * proximity
        rescored.append((e, blended))

    rescored.sort(key=lambda x: -x[1])
    return [e for e, _ in rescored]
