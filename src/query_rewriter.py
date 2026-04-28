"""
Query rewriter for Claude Total Memory v8 retrieval pipeline.

Calls Anthropic Haiku to produce three query variants for multi-query RRF
fusion on LoCoMo-style long-term memory benchmarks:

  - canonical  : single-fact lookup reformulation
  - decomposed : 1-3 sub-queries for multi-hop questions
  - hyde       : hypothetical answer for HyDE-style embedding expansion

Public API:
  - rewrite(query, client=None, model=...) -> dict        (LRU-cached)
  - expand_for_retrieval(query, client=None, model=...) -> list[str]
  - has_decomposable_intent(query) -> bool                (cheap heuristic)
  - is_enabled() -> bool                                  (env toggle)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import lru_cache
from typing import Any

__all__ = [
    "rewrite",
    "expand_for_retrieval",
    "has_decomposable_intent",
    "is_enabled",
]

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 200
MAX_ATTEMPTS = 3

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You rewrite user questions for a memory retrieval system. "
    "Output ONE single-line minified JSON object with keys "
    '"canonical" (string), "decomposed" (array of 0-3 strings), '
    '"hyde" (1-2 sentence hypothetical answer string). '
    "Drop conversational filler. Decompose only true multi-hop questions; "
    "single-fact lookups get decomposed=[]. No markdown, no prose, JSON only.\n"
    "Example 1:\n"
    'Q: "Hey, do you remember when Alice moved to Berlin?"\n'
    'A: {"canonical":"Alice move to Berlin date","decomposed":[],'
    '"hyde":"Alice moved to Berlin in 2022."}\n'
    "Example 2:\n"
    'Q: "Compare what Bob bought in March and what Carol bought in April"\n'
    'A: {"canonical":"Bob March purchases vs Carol April purchases",'
    '"decomposed":["Bob purchases in March","Carol purchases in April"],'
    '"hyde":"Bob bought a laptop in March; Carol bought a camera in April."}\n'
    "Example 3:\n"
    'Q: "Когда Иван переехал и куда именно?"\n'
    'A: {"canonical":"Иван дата и место переезда",'
    '"decomposed":["Иван дата переезда","Иван место переезда"],'
    '"hyde":"Иван переехал в Москву в 2021 году."}'
)

# Heuristic markers for multi-hop intent.
_EN_MULTI_MARKERS = (
    " and ", " both ", " compare", "vs ", " versus ",
    " after ", " before ", " between ",
    " as well as ", " along with ",
)
_RU_MULTI_MARKERS = (
    " и ", " а также", "в том числе", " сравни", " до ", " после ",
    " между ", " вместе с ",
)


def _log(msg: str) -> None:
    sys.stderr.write(f"[query_rewriter] {msg}\n")


def is_enabled() -> bool:
    """Return True when MEMORY_QUERY_REWRITE env var is set to '1'/'true'."""
    return os.environ.get("MEMORY_QUERY_REWRITE", "0").lower() in {"1", "true", "yes", "on"}


def has_decomposable_intent(query: str) -> bool:
    """Cheap heuristic gate: does this query look multi-hop / compound?

    Triggers on conjunctions, comparison markers, temporal ranges, and
    question-form commas in EN/RU. False positives are fine — the LLM
    returns decomposed=[] when it decides the query is single-fact.
    """
    if not query or len(query) < 8:
        return False
    q = query.lower()

    for m in _EN_MULTI_MARKERS:
        if m in q:
            return True
    for m in _RU_MULTI_MARKERS:
        if m in q:
            return True

    # "X, Y?" style question with comma + question mark
    if "?" in q and "," in q:
        return True

    # Multiple question words in one query (what + when / что + когда)
    qwords = ("what", "when", "where", "who", "why", "how",
              "что", "когда", "где", "кто", "почему", "как")
    hits = sum(1 for w in qwords if re.search(rf"\b{w}\b", q))
    if hits >= 2:
        return True

    return False


def _fallback(query: str) -> dict[str, Any]:
    return {"canonical": query, "decomposed": [], "hyde": ""}


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = _FENCE_RE.sub("", text).strip()
    # If Haiku wrapped JSON in prose, extract first {...} block
    if not text.startswith("{"):
        m = _JSON_OBJ_RE.search(text)
        if m:
            text = m.group(0)
    return text


def _parse_response(raw: str, query: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        _log(f"JSON parse failed: {e}; raw={raw[:200]!r}")
        return _fallback(query)

    canonical = data.get("canonical") or query
    decomposed = data.get("decomposed") or []
    hyde = data.get("hyde") or ""

    # Coerce + sanitize
    if not isinstance(canonical, str):
        canonical = query
    if not isinstance(decomposed, list):
        decomposed = []
    decomposed = [str(x).strip() for x in decomposed if isinstance(x, (str, int, float)) and str(x).strip()]
    decomposed = decomposed[:3]
    if not isinstance(hyde, str):
        hyde = ""

    return {
        "canonical": canonical.strip() or query,
        "decomposed": decomposed,
        "hyde": hyde.strip(),
    }


def _get_client(client: Any = None) -> Any:
    if client is not None:
        return client
    import anthropic  # lazy import
    return anthropic.Anthropic()


def _call_haiku(query: str, client: Any, model: str) -> dict[str, Any]:
    """Call Haiku with retries. Returns parsed dict or fallback on failure."""
    # v11 Phase 5 — defensive telemetry on the Anthropic SDK code path
    # (which doesn't go through llm_provider._http_post_json).
    try:
        from memory_core.telemetry import counters as _v11_counters
        _v11_counters.bump("llm_calls", 1.0)
        _v11_counters.bump("network_calls", 1.0)
    except Exception:
        pass
    last_err: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            # Extract text content
            parts = getattr(resp, "content", []) or []
            text = ""
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    text += t
            if not text:
                raise ValueError("empty response from Haiku")
            return _parse_response(text, query)
        except Exception as e:
            last_err = e
            backoff = 2 ** attempt  # 1, 2, 4
            _log(f"attempt {attempt + 1}/{MAX_ATTEMPTS} failed: {e}; sleeping {backoff}s")
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(backoff)

    _log(f"all retries exhausted: {last_err}")
    return _fallback(query)


@lru_cache(maxsize=512)
def _rewrite_cached(query: str, model: str) -> tuple[str, str, tuple[str, ...]]:
    """Internal cached worker. Returns tuple (cached-friendly) converted by rewrite()."""
    try:
        client = _get_client(None)
    except Exception as e:
        _log(f"client init failed: {e}")
        return (query, "", ())
    result = _call_haiku(query, client, model)
    return (
        result["canonical"],
        result["hyde"],
        tuple(result["decomposed"]),
    )


def rewrite(
    query: str,
    client: Any = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Rewrite a user query into canonical / decomposed / hyde variants.

    Cached on (query, model). Never raises — returns fallback dict on any
    failure. Pass an explicit ``client`` to bypass the LRU cache (useful
    for tests with a mock client).
    """
    if not query or not query.strip():
        return _fallback(query)

    # Explicit client bypasses cache (otherwise we'd leak a client reference
    # into the hashed key or lose test-time control).
    if client is not None:
        return _call_haiku(query, client, model)

    canonical, hyde, decomposed = _rewrite_cached(query, model)
    return {
        "canonical": canonical,
        "decomposed": list(decomposed),
        "hyde": hyde,
    }


def expand_for_retrieval(
    query: str,
    client: Any = None,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Return [original, canonical, hyde] deduplicated for multi-query RRF.

    Empty strings are dropped. Order preserves priority (original first).
    """
    if not query or not query.strip():
        return []

    r = rewrite(query, client=client, model=model)
    candidates = [query, r.get("canonical", ""), r.get("hyde", "")]

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = (c or "").strip()
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


if __name__ == "__main__":
    # Smoke test: three queries — single-hop EN, multi-hop EN, Russian multi-hop.
    samples = [
        "When did Alice move to Berlin?",
        "Compare what Bob bought in March and what Carol bought in April",
        "Когда Иван переехал и куда именно?",
    ]

    print(f"MEMORY_QUERY_REWRITE={os.environ.get('MEMORY_QUERY_REWRITE', '0')}")
    print(f"is_enabled() = {is_enabled()}")
    print(f"ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    print("-" * 60)

    for q in samples:
        print(f"\nQ: {q}")
        print(f"  has_decomposable_intent = {has_decomposable_intent(q)}")
        r = rewrite(q)
        print(f"  canonical : {r['canonical']}")
        print(f"  decomposed: {r['decomposed']}")
        print(f"  hyde      : {r['hyde']}")
        print(f"  expand    : {expand_for_retrieval(q)}")
