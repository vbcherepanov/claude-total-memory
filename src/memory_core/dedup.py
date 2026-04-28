"""v11.0 Phase 3 — Deterministic dedup helpers.

Two cheap operations that run in the save hot path:

* :func:`exact_dedup` — returns (sha256 of raw content, sha256 of
  normalized content). The normalized hash collapses whitespace, lowercases
  and strips so trivially different inputs share a key.
* :func:`find_duplicate` — wraps the legacy `Store._find_duplicate` which
  combines FTS5 lookup with Jaccard / fuzzy-ratio scoring. The wrapper
  isolates that call site so future replacements don't need to touch
  every save path.

No LLM, no network. Pure local logic.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional


_RE_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace runs to single spaces."""
    if not text:
        return ""
    return _RE_WS.sub(" ", text.strip().lower())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_dedup(content: str) -> tuple[str, str]:
    """Return (raw_sha256, normalized_sha256) for the given content."""
    return _sha256(content or ""), _sha256(normalize(content or ""))


def find_duplicate(
    db_or_store: Any,
    content: str,
    ktype: str,
    project: str,
) -> Optional[int]:
    """Look up a near-duplicate via the legacy `Store._find_duplicate`.

    Accepts either a `Store` instance (preferred — gives access to the
    full FTS+Jaccard ladder) or a raw sqlite connection (falls back to a
    minimal FTS-only check).
    """
    # Preferred: full store-level dedup.
    finder = getattr(db_or_store, "_find_duplicate", None)
    if callable(finder):
        try:
            return finder(content, ktype, project)
        except Exception:  # noqa: BLE001 — never let dedup raise into save
            return None

    # Fallback: ad-hoc FTS5 nearest-match if we only have a connection.
    try:
        words = [w for w in (content or "").split()[:10] if len(w) > 2]
        if not words:
            return None
        fts_q = " OR ".join(
            re.sub(r'[^a-zA-Z0-9_]+', "", w) or w for w in words
        )
        rows = db_or_store.execute(
            """
            SELECT k.id FROM knowledge_fts f
            JOIN knowledge k ON k.id = f.rowid
            WHERE f.content MATCH ? AND k.status='active'
              AND k.project=? AND k.type=?
            ORDER BY rank LIMIT 1
            """,
            (fts_q, project, ktype),
        ).fetchall()
        if rows:
            return int(rows[0][0])
    except Exception:  # noqa: BLE001
        pass
    return None


__all__ = ["normalize", "exact_dedup", "find_duplicate"]
