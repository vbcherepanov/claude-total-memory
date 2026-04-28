"""Keyword extraction shim around `representations.generate_representations`.

The underlying LLM returns a comma-separated string; this shim parses it
into a list, deduplicates case-insensitively, preserves first-seen order,
and trims the result to `max_n`.

Same warning as the other LLM helpers: this is async-tier work, not for
the save/search hot path.
"""

from __future__ import annotations

import representations as _representations


def extract_keywords(text: str, max_n: int = 10) -> list[str]:
    """Return up to `max_n` keywords for `text`.

    Returns `[]` when the underlying LLM is unavailable, the call fails,
    or the response is empty. Never raises.
    """
    if max_n <= 0:
        return []

    bundle = _representations.generate_representations(
        text or "",
        skip={"summary", "questions", "compressed"},
    )
    raw = (bundle.get("keywords") or "").strip()
    if not raw:
        return []

    seen: dict[str, None] = {}
    out: list[str] = []
    # Representations prompt asks for a comma-separated list, but be
    # liberal in what we accept: also split on newlines.
    for token in raw.replace("\n", ",").split(","):
        kw = token.strip().strip("•-*").strip()
        if not kw:
            continue
        key = kw.lower()
        if key in seen:
            continue
        seen[key] = None
        out.append(kw)
        if len(out) >= max_n:
            break
    return out


__all__ = ["extract_keywords"]
