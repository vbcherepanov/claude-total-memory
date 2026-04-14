"""Adaptive verbosity — pick the appropriate `detail` level from query shape.

When `memory_recall` is called without explicit `detail`, inspect the query
and choose:
  - compact  — short, keyword-style queries (< 4 words, no structure)
  - summary  — medium natural-language queries
  - full     — queries with paths, URLs, code symbols, line numbers, or
               multiple clauses (user needs everything)

No LLM hops — pure heuristic. Cheap to run in the hot path.
"""

from __future__ import annotations

import re

# Pre-compiled signal patterns
_PATH_RE = re.compile(r"(?:^|\s)[/~][^\s]*")
_URL_RE = re.compile(r"https?://\S+")
_LINE_RE = re.compile(r"\bline\s+\d+|\b:\d{2,}\b")
_CODE_SYM_RE = re.compile(r"\w+\(\)|::")
_MULTI_CLAUSE_RE = re.compile(r"\band\b.+\band\b", re.IGNORECASE)


def analyze_query_complexity(query: str) -> str:
    """Return 'compact', 'summary' or 'full' based on query shape."""
    q = (query or "").strip()
    if not q:
        return "compact"

    # Full-detail signals — user wants everything
    if _PATH_RE.search(q):
        return "full"
    if _URL_RE.search(q):
        return "full"
    if _LINE_RE.search(q):
        return "full"
    if _CODE_SYM_RE.search(q):
        return "full"
    if _MULTI_CLAUSE_RE.search(q):
        return "full"

    words = q.split()
    n = len(words)
    if n >= 10:
        return "full"
    if n <= 3:
        return "compact"
    return "summary"
