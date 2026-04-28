"""Utility-question generation shim around `representations.generate_representations`.

GEM-RAG-style "what questions does this record answer?" probes — useful
for question-aware retrieval. The underlying LLM returns a numbered or
bulleted list as a single string; this shim parses, normalises, and caps.

Async-tier code. Never call from the save/search hot path.
"""

from __future__ import annotations

import re

import representations as _representations

# Strip the leading "1.", "1)", "•", "-", "*", "Q1:" decorations that the
# model tends to emit on questions lists.
_LEADING_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|q\d+[:.])\s*", re.IGNORECASE)


def generate_questions(text: str, max_n: int = 5) -> list[str]:
    """Return up to `max_n` utility questions answered by `text`.

    Returns `[]` if the LLM is unavailable, the call fails, or the response
    is empty. Never raises.
    """
    if max_n <= 0:
        return []

    bundle = _representations.generate_representations(
        text or "",
        skip={"summary", "keywords", "compressed"},
    )
    raw = (bundle.get("questions") or "").strip()
    if not raw:
        return []

    out: list[str] = []
    for line in raw.splitlines():
        q = _LEADING_BULLET.sub("", line).strip().strip('"').strip()
        if not q:
            continue
        # The prompt asks for questions but the model occasionally drops
        # the trailing question mark — normalise.
        if not q.endswith("?"):
            q += "?"
        out.append(q)
        if len(out) >= max_n:
            break
    return out


__all__ = ["generate_questions"]
