"""Re-export of `src/coref_resolver.py` (A1 of the hot-path audit).

Resolves pronouns/deictics ("this", "the previous one") in incoming
content against recent session history via an LLM round-trip. Disabled
by default in v11 fast mode and runs through the async enrichment
worker when re-enabled.
"""

from __future__ import annotations

from coref_resolver import (  # noqa: F401  (re-exports)
    CorefResult,
    needs_resolution,
    resolve,
)

__all__ = [
    "CorefResult",
    "needs_resolution",
    "resolve",
]
