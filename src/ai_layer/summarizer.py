"""Single-purpose summary helper around `representations.generate_representations`.

`generate_representations` produces summary/keywords/questions/compressed in
one LLM round-trip. v11 callers that only need the summary should not have
to know about that bundle — this shim hides it.

The function is an LLM round-trip and therefore MUST NOT be called from
`memory_core` or any sync save/search hot path. Use it from the async
worker (`ai_layer.enrichment_worker`) or from offline tooling only.
"""

from __future__ import annotations

import representations as _representations


def summarize(text: str, *, target_chars: int = 200) -> str:
    """Return a single-sentence summary of `text`.

    Returns the empty string when:
      - text is below the LLM threshold (~50 tokens),
      - the configured representations LLM is unavailable,
      - the LLM call fails (network / model / parse error).

    `target_chars` is advisory; the summary may be slightly longer or
    shorter depending on the underlying prompt budget.
    """
    bundle = _representations.generate_representations(
        text or "",
        skip={"keywords", "questions", "compressed"},
    )
    summary = (bundle.get("summary") or "").strip()
    if target_chars > 0 and len(summary) > target_chars:
        # Hard cap for callers with a strict UI budget.
        summary = summary[: target_chars - 1].rstrip() + "…"
    return summary


__all__ = ["summarize"]
