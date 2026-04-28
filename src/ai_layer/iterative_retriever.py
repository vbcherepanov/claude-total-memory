"""Iterative retrieval (IRCoT-style) for multi-hop LoCoMo queries.

A single retrieval pass can't follow chains of 2-3 facts. This module
implements an iterative loop:

    decompose query -> retrieve sub-query -> partial answer ->
    derive next sub-query -> retrieve again, up to N iterations.

The decomposer is the existing ``query_rewriter.rewrite()`` (Anthropic
Haiku). Each iteration calls a planner LLM that, given the original
question + evidence so far + partial answers, decides whether more
retrieval is needed and emits the next sub-query.

Public API
----------

    iterative_retrieve(query, *, search_fn, project=None,
                       max_iters=4, k_per_iter=5,
                       llm_model="haiku", llm_client=None) -> IterativeResult

The ``search_fn`` callable is injected so this module stays decoupled
from ``memory_core.recall`` (the import wall in v11). ``llm_client`` is
also injectable, with a small protocol so tests can pass a fake without
spinning up real Anthropic / OpenAI SDKs.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

# `query_rewriter` lives under ``src/`` and is re-exported via ``ai_layer``.
# Import the canonical module so the LRU cache is shared across the codebase.
from query_rewriter import rewrite as _rewrite

__all__ = [
    "IterativeResult",
    "PlannerDecision",
    "LLMClientProtocol",
    "SearchFn",
    "iterative_retrieve",
]

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Protocols & dataclasses
# ──────────────────────────────────────────────────────────────────────


class SearchFn(Protocol):
    """Retrieval callable injected by the caller (typically Recall.search)."""

    def __call__(  # pragma: no cover - protocol signature
        self,
        query: str,
        k: int = 10,
        project: str | None = None,
    ) -> list[dict[str, Any]]: ...


class LLMClientProtocol(Protocol):
    """Minimal contract the planner needs from an LLM client.

    ``benchmarks._llm_adapter.LLMClient`` satisfies this directly. Tests
    pass a tiny fake whose ``complete()`` returns queued strings.
    """

    def complete(  # pragma: no cover - protocol signature
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retries: int = 3,
    ) -> Any: ...


@dataclass
class PlannerDecision:
    """One iteration's planner output."""

    partial_answer: str
    next_query: str | None
    done: bool
    raw: str = ""
    parse_attempts: int = 1


@dataclass
class IterativeResult:
    """Outcome of an :func:`iterative_retrieve` invocation."""

    final_evidence: list[dict[str, Any]]
    sub_queries: list[str]
    partial_answers: list[str]
    iterations_used: int
    terminated_reason: str  # "converged" | "max_iters" | "decomposer_empty"
    provenance: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# LLM prompt
# ──────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = (
    "You are an iterative retrieval planner. Given a user question, "
    "evidence retrieved so far, and partial answers, decide if more "
    "retrieval is needed. Output ONE minified JSON object: "
    '{"partial_answer": str, "next_query": str|null, "done": bool}. '
    "Set done=true when evidence fully answers the question. "
    "next_query=null only when done=true."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _hit_id(hit: dict[str, Any]) -> str:
    """Stable identity for an evidence hit.

    Prefers explicit ``id``; falls back to ``rowid`` then a content hash
    so dedup still works for hits without IDs (e.g. graph triples).
    """
    for key in ("id", "rowid", "node_id", "fact_id"):
        v = hit.get(key)
        if v is not None:
            return f"{key}:{v}"
    content = str(hit.get("content", ""))[:256]
    return f"sha:{hash(content)}"


def _format_evidence(evidence: list[dict[str, Any]], limit: int = 12) -> str:
    """Render evidence as a numbered list for the planner prompt.

    Truncates to ``limit`` items (most recent) and content to 280 chars
    to keep token cost bounded across iterations.
    """
    if not evidence:
        return "(none yet)"
    tail = evidence[-limit:]
    lines: list[str] = []
    offset = len(evidence) - len(tail)
    for idx, hit in enumerate(tail, start=1):
        content = str(hit.get("content", "")).strip().replace("\n", " ")
        if len(content) > 280:
            content = content[:277] + "..."
        lines.append(f"[{offset + idx}] {content}")
    return "\n".join(lines)


def _format_partial_answers(answers: list[str]) -> str:
    if not answers:
        return "(none yet)"
    return "\n".join(f"- {a}" for a in answers if a)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    if not text.startswith("{"):
        m = _JSON_OBJ_RE.search(text)
        if m:
            text = m.group(0)
    return text


def _parse_planner_response(raw: str) -> dict[str, Any]:
    """Strict JSON parse. Raises ``ValueError`` on malformed input."""
    cleaned = _strip_fences(raw)
    if not cleaned:
        raise ValueError("empty planner response")
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError(f"planner returned non-object: {type(data).__name__}")

    partial = data.get("partial_answer", "")
    next_q = data.get("next_query", None)
    done_raw = data.get("done", False)

    if not isinstance(partial, str):
        partial = str(partial)
    if next_q is not None and not isinstance(next_q, str):
        next_q = str(next_q)
    if isinstance(next_q, str):
        next_q = next_q.strip() or None
    done = bool(done_raw)

    # Contract: next_query=null only when done=true. If the model violates
    # this we coerce to the safer interpretation (treat as done) rather than
    # spin another retrieval on a null query.
    if next_q is None and not done:
        done = True

    return {"partial_answer": partial.strip(), "next_query": next_q, "done": done}


def _extract_text(resp: Any) -> str:
    """Pull plain text out of any LLMResult-like object or a raw string."""
    if isinstance(resp, str):
        return resp
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        return text
    # Some adapters return an object with .content[0].text (Anthropic SDK).
    content = getattr(resp, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        t = getattr(first, "text", None)
        if isinstance(t, str):
            return t
    return str(resp or "")


def _call_planner(
    llm_client: LLMClientProtocol,
    *,
    model: str,
    question: str,
    evidence: list[dict[str, Any]],
    partial_answers: list[str],
) -> PlannerDecision:
    """Invoke the planner LLM with one retry on JSON parse failure."""
    user_prompt = (
        f"QUESTION: {question}\n"
        f"EVIDENCE:\n{_format_evidence(evidence)}\n"
        f"PARTIAL_ANSWERS_SO_FAR:\n{_format_partial_answers(partial_answers)}\n"
        "Return JSON."
    )

    last_err: Exception | None = None
    last_raw = ""
    for attempt in range(1, 3):  # one retry
        try:
            resp = llm_client.complete(
                system=PLANNER_SYSTEM_PROMPT,
                user=user_prompt,
                model=model,
                max_tokens=256,
                temperature=0.0,
            )
        except Exception as e:  # noqa: BLE001 — we surface as graceful stop
            last_err = e
            log.warning("iterative planner LLM call failed (attempt %d): %s", attempt, e)
            break

        last_raw = _extract_text(resp)
        try:
            parsed = _parse_planner_response(last_raw)
            return PlannerDecision(
                partial_answer=parsed["partial_answer"],
                next_query=parsed["next_query"],
                done=parsed["done"],
                raw=last_raw,
                parse_attempts=attempt,
            )
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            log.warning(
                "iterative planner JSON parse failed (attempt %d): %s; raw=%r",
                attempt,
                e,
                last_raw[:200],
            )

    # Both attempts failed — terminate gracefully with what we have.
    return PlannerDecision(
        partial_answer="",
        next_query=None,
        done=True,
        raw=last_raw,
        parse_attempts=2 if last_err is not None else 1,
    )


def _seed_sub_queries(query: str, llm_client: LLMClientProtocol | None) -> tuple[list[str], dict[str, Any]]:
    """Use ``query_rewriter.rewrite`` to seed the sub-query queue.

    Returns ``(sub_queries, rewrite_meta)``. Falls back to ``[query]`` if
    rewrite is unavailable or returns empty decomposition.
    """
    meta: dict[str, Any] = {"used_decomposition": False}
    try:
        rewrite_kwargs: dict[str, Any] = {}
        if llm_client is not None:
            # Passing client bypasses the LRU cache — fine for tests.
            rewrite_kwargs["client"] = llm_client
        r = _rewrite(query, **rewrite_kwargs)
    except Exception as e:  # noqa: BLE001
        log.warning("query_rewriter.rewrite failed: %s; falling back to canonical", e)
        return [query], meta

    decomposed = [q for q in (r.get("decomposed") or []) if isinstance(q, str) and q.strip()]
    canonical = (r.get("canonical") or query).strip() or query
    meta["canonical"] = canonical
    meta["decomposed"] = list(decomposed)

    if decomposed:
        meta["used_decomposition"] = True
        return decomposed, meta
    return [canonical], meta


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def iterative_retrieve(
    query: str,
    *,
    search_fn: SearchFn,
    project: str | None = None,
    max_iters: int = 4,
    k_per_iter: int = 5,
    llm_model: str = "haiku",
    llm_client: LLMClientProtocol | None = None,
) -> IterativeResult:
    """Run an IRCoT-style iterative retrieval loop.

    Parameters
    ----------
    query
        User question. Multi-hop questions benefit most.
    search_fn
        Retrieval callable. Must accept ``(query, k, project)`` and
        return a list of dicts with ``id`` (or ``rowid``) and ``content``.
    project
        Optional project filter forwarded to ``search_fn``.
    max_iters
        Hard cap on retrieval rounds. Each round = 1 search + 1 planner LLM.
    k_per_iter
        Top-K to fetch per round.
    llm_model
        Alias accepted by the LLM adapter (``haiku``, ``sonnet``, ``gpt-4o``...).
    llm_client
        Injectable client. If ``None``, constructs a ``benchmarks._llm_adapter.LLMClient``
        on first use.

    Returns
    -------
    IterativeResult
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if k_per_iter < 1:
        raise ValueError("k_per_iter must be >= 1")

    started = time.perf_counter()

    # Lazy-construct the LLM client so callers don't pay for it when only
    # search_fn is exercised (and tests can always inject a fake).
    if llm_client is None:
        from benchmarks._llm_adapter import LLMClient  # noqa: PLC0415

        llm_client = LLMClient(provider="auto", default_model=llm_model)

    sub_queries_seed, rewrite_meta = _seed_sub_queries(query, llm_client=None)
    pending: list[str] = list(sub_queries_seed)

    issued: list[str] = []
    partial_answers: list[str] = []
    evidence: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    per_iter: list[dict[str, Any]] = []

    terminated_reason = "max_iters"

    if not pending:
        return IterativeResult(
            final_evidence=[],
            sub_queries=[],
            partial_answers=[],
            iterations_used=0,
            terminated_reason="decomposer_empty",
            provenance={
                "rewrite": rewrite_meta,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "iters": [],
            },
        )

    iters_used = 0
    for _ in range(max_iters):
        if not pending:
            terminated_reason = "converged"
            break

        sub_q = pending.pop(0)
        iters_used += 1
        issued.append(sub_q)

        iter_started = time.perf_counter()
        try:
            hits = search_fn(sub_q, k=k_per_iter, project=project) or []
        except Exception as e:  # noqa: BLE001
            log.warning("search_fn failed on sub-query %r: %s", sub_q, e)
            hits = []

        new_ids: list[str] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            hid = _hit_id(hit)
            if hid in seen_ids:
                continue
            seen_ids.add(hid)
            evidence.append(hit)
            new_ids.append(hid)

        decision = _call_planner(
            llm_client,
            model=llm_model,
            question=query,
            evidence=evidence,
            partial_answers=partial_answers,
        )
        if decision.partial_answer:
            partial_answers.append(decision.partial_answer)

        per_iter.append(
            {
                "iter": iters_used,
                "sub_query": sub_q,
                "hits_returned": len(hits),
                "new_evidence_ids": new_ids,
                "planner_done": decision.done,
                "planner_next_query": decision.next_query,
                "planner_parse_attempts": decision.parse_attempts,
                "elapsed_ms": (time.perf_counter() - iter_started) * 1000.0,
            }
        )

        if decision.done or decision.next_query is None:
            terminated_reason = "converged"
            break

        # Push planner-suggested next query to the front (LIFO for the
        # follow-up so it runs before any leftover decomposed seeds).
        pending.insert(0, decision.next_query)

    provenance = {
        "rewrite": rewrite_meta,
        "iters": per_iter,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "evidence_count": len(evidence),
    }

    return IterativeResult(
        final_evidence=evidence,
        sub_queries=issued,
        partial_answers=partial_answers,
        iterations_used=iters_used,
        terminated_reason=terminated_reason,
        provenance=provenance,
    )
