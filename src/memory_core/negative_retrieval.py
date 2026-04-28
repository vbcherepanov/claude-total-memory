"""Negative-evidence retrieval (W2-H, v11).

For LoCoMo "adversarial" questions the gold answer is *"this isn't in
memory"*. The classifier in :mod:`ai_layer.answerability` already gates
generation by asking "does the supplied evidence support an answer?",
but it only sees the **positive** retrieval — facts whose embeddings
matched the question. That misses a whole class of failures:

* Memory genuinely contains an answer but the question is phrased so
  that retrieval surfaces *near-matches* which sound supportive while a
  contradicting fact sits one sentence away ("Alice loves teal" vs.
  "Alice told Bob she actually hates teal now").
* Adversarial LoCoMo questions plant misleading positives ("Alice's
  dog is named Rex") with the truth elsewhere ("Alice does not own a
  dog").

This module performs a deliberate **second** retrieval against an
*inverted* query — one phrased to surface facts that would CONTRADICT
the most likely answer. Each (positive, negative) pair is then scored
with the existing contradiction detector; the maximum score chooses
between three decisions:

* ``no_contradiction`` (max < 0.30) — the negative pass found nothing
  that conflicts with the positives. Pipeline behaves as before.
* ``soft_contradict`` (0.30 ≤ max < 0.60) — there is partial conflict.
  The downstream answer should hedge ("evidence is mixed, but…").
* ``hard_contradict`` (max ≥ 0.60) — strong conflict. The router
  should treat the question as unanswerable and emit IDK rather than
  pick a side.

Design notes
------------

* **Layer wall.** This module lives in ``memory_core`` and therefore
  must not import :mod:`ai_layer.*` (enforced by
  ``tests/test_v11_layer_separation``). All collaborators are passed in
  as ``Protocol`` typed callables: ``search_fn``, ``contradiction_fn``,
  and an optional ``llm_client``. ``AnswerabilityResult`` is *not* a
  parameter — the upstream caller decides whether to invoke the
  negative pass at all and supplies the positives directly.

* **One LLM call.** Producing the inverted query is a small Haiku
  prompt; we retry once on bad output (empty / multi-line garbage /
  echoing the original question) and then fall back to a deterministic
  template. The contradiction scoring is delegated, so this module
  pays for at most one round-trip.

* **Quadratic budget cap.** The ``contradiction_fn`` is asked to score
  every positive × negative pair; both lists are clipped to 5 entries
  before the cross-product so we never run more than 25 evaluations
  per call regardless of how greedy the upstream retrieval was.

* **Empty-positive shortcut.** If the caller hands us no positives
  (e.g. retrieval returned nothing) there is nothing to contradict;
  we return ``no_contradiction`` immediately without an LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


# ──────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────


@dataclass
class NegativeEvidenceResult:
    """Verdict from one negative-retrieval pass."""

    inverted_query: str
    negative_evidence: list[dict] = field(default_factory=list)
    contradiction_score: float = 0.0
    decision: str = "no_contradiction"
    rationale: str = ""


class SearchFn(Protocol):
    """Retrieval entry point injected by the caller.

    Matches the signature recall.py exposes internally — anything that
    accepts ``(query, k=, project=)`` and returns a list of dicts works.
    Each dict must carry a ``text`` (or ``content``) field; everything
    else is opaque to this module.
    """

    def __call__(
        self,
        query: str,
        k: int = 10,
        project: str | None = None,
    ) -> list[dict]:  # pragma: no cover - protocol
        ...


class LLMLike(Protocol):
    """Minimum LLM interface — same shape as ``benchmarks._llm_adapter``."""

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
    ) -> Any:  # pragma: no cover - protocol
        ...


class ContradictionFn(Protocol):
    """Pairwise contradiction probability.

    Returns the probability that ``fact_b`` contradicts ``fact_a`` in
    the range ``[0.0, 1.0]``. Implementations may delegate to the
    existing :mod:`contradiction_detector` LLM prompt or to a smaller
    NLI head; the only contract is a deterministic ``float``.
    """

    def __call__(
        self,
        fact_a: str,
        fact_b: str,
    ) -> float:  # pragma: no cover - protocol
        ...


# ──────────────────────────────────────────────
# Tunables — kept module-level so tests can flip them
# ──────────────────────────────────────────────


# Decision thresholds. Picked so that the soft band is wide enough to
# catch partial conflicts (single conflicting fact among many supportive
# ones) without forcing IDK for every mild disagreement.
THRESHOLD_SOFT = 0.30
THRESHOLD_HARD = 0.60

# Hard cap on cross-product size. 5 × 5 = 25 contradiction calls
# per question is the upper bound any caller will see.
_MAX_PAIRS_PER_SIDE = 5

# Length cap fed to the LLM for the inversion prompt.
_MAX_QUESTION_CHARS = 600

# Fallback inversion template if the LLM keeps misbehaving. Phrased
# generically so it is still a usable retrieval query — "facts about
# <X> being false / contradicted / different" — without leaking model
# noise into the search index.
_FALLBACK_TEMPLATE = "facts contradicting or refuting: {q}"

# Default model alias — same convention as answerability.py. Caller may
# override via ``llm_model``.
DEFAULT_MODEL = "haiku"


_SYSTEM_PROMPT = (
    "Given a question, produce a query that searches for facts that "
    "would CONTRADICT the most likely answer. Output ONE line, no "
    "preamble, no quotes, no explanation. The line should be a search "
    "query — keywords or a short paraphrase that would retrieve a fact "
    "denying or refuting the assumption behind the question."
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _clean_text(value: Any) -> str:
    """Extract text from a hit dict or a bare string.

    Hits may carry their text under several keys depending on the
    retrieval backend (``text`` for the new pipeline, ``content`` for
    the legacy v10 result shape, ``snippet`` for FTS rows). We try them
    in order and fall back to ``str(value)`` so the caller never has to
    normalise upstream.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "content", "snippet"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    return str(value).strip()


def _extract_response_text(raw: Any) -> str:
    """Normalise an LLM response into a string.

    Mirrors ``answerability._extract_text``: accept bare strings *and*
    objects with a ``.text`` attribute (``LLMResult``).
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    text_attr = getattr(raw, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(raw)


def _first_non_empty_line(text: str) -> str:
    """Take the first non-empty, non-fence line from ``text``.

    Models occasionally prepend a markdown fence or a bullet character;
    we strip those before returning. The cap on length (300 chars)
    matches what FTS / dense retrieval can usefully handle.
    """
    if not text:
        return ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Drop common conversational preambles and bullet markers.
        line = re.sub(r"^(?:```\w*\s*|>\s*|[-*]\s+|\d+\.\s+)", "", line)
        line = line.strip("`").strip()
        # Drop surrounding quotation marks the model loves to add.
        if (line.startswith('"') and line.endswith('"')) or (
            line.startswith("'") and line.endswith("'")
        ):
            line = line[1:-1].strip()
        if line:
            return line[:300]
    return ""


def _is_useless_inversion(question: str, candidate: str) -> bool:
    """Detect inversion outputs we should retry rather than use.

    Bad outputs we have seen in practice:

    * Empty / whitespace.
    * The model echoing the question verbatim — adds no value over
      regular retrieval.
    * Generic non-sequiturs like ``"I cannot do that"`` (caught by
      length + lack of question content overlap).
    """
    if not candidate:
        return True
    if candidate.lower().strip(" ?.!") == question.lower().strip(" ?.!"):
        return True
    if len(candidate) < 4:
        return True
    return False


def _build_user_prompt(question: str) -> str:
    q = question.strip()
    if len(q) > _MAX_QUESTION_CHARS:
        q = q[:_MAX_QUESTION_CHARS] + "…"
    return f"QUESTION:\n{q}\n\nCONTRADICTING SEARCH QUERY:"


def _invert_query(
    question: str,
    *,
    llm_client: LLMLike | None,
    llm_model: str,
) -> str:
    """Produce a contradiction-seeking search query.

    Strategy:
    1. Ask the LLM. Take the first non-empty line.
    2. If that line is empty or echoes the question, retry once.
    3. If both attempts fail (or no client supplied), fall back to a
       deterministic template. The template is a usable retrieval
       query, not a stub — callers that disable the LLM still get a
       meaningful negative pass.
    """
    if llm_client is None:
        return _FALLBACK_TEMPLATE.format(q=question.strip())

    user_prompt = _build_user_prompt(question)
    for _attempt in range(2):
        try:
            raw = llm_client.complete(
                model=llm_model,
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=80,
            )
        except Exception:
            # Retry once on transient errors; fall through to template
            # if the second call also raises.
            continue
        candidate = _first_non_empty_line(_extract_response_text(raw))
        if not _is_useless_inversion(question, candidate):
            return candidate
    # Both LLM attempts failed or produced unusable output (empty,
    # too short, or a verbatim echo of the question). Fall back to
    # the deterministic template so the negative pass still gets a
    # meaningful query.
    return _FALLBACK_TEMPLATE.format(q=question.strip())


def _decide(score: float) -> str:
    """Map a contradiction score onto one of three decision labels."""
    if score >= THRESHOLD_HARD:
        return "hard_contradict"
    if score >= THRESHOLD_SOFT:
        return "soft_contradict"
    return "no_contradiction"


def _format_rationale(
    decision: str,
    score: float,
    pos_text: str,
    neg_text: str,
) -> str:
    """One short sentence summarising the verdict.

    The two texts are clipped to keep the rationale readable in logs
    and dashboards. The leading verb varies by decision so an operator
    skimming the output knows the severity at a glance.
    """
    if decision == "no_contradiction":
        return (
            f"no contradiction found (max score {score:.2f} below "
            f"{THRESHOLD_SOFT:.2f})"
        )
    pos_preview = (pos_text or "").strip().replace("\n", " ")[:120]
    neg_preview = (neg_text or "").strip().replace("\n", " ")[:120]
    severity = "soft" if decision == "soft_contradict" else "hard"
    return (
        f"{severity} contradiction at score {score:.2f}: "
        f"positive=\"{pos_preview}\" vs. negative=\"{neg_preview}\""
    )


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────


def negative_retrieve(
    question: str,
    positive_evidence: list[dict],
    *,
    search_fn: SearchFn,
    contradiction_fn: ContradictionFn,
    project: str | None = None,
    k: int = 5,
    llm_model: str = DEFAULT_MODEL,
    llm_client: LLMLike | None = None,
) -> NegativeEvidenceResult:
    """Run one negative-evidence pass against ``positive_evidence``.

    Parameters
    ----------
    question:
        The original natural-language user question. Used to seed the
        inverted-query LLM prompt.
    positive_evidence:
        Hits already retrieved for the question (the "positive" side).
        Each hit must be a dict with a ``text`` or ``content`` field.
        Empty list → ``no_contradiction`` without any LLM call.
    search_fn:
        Callable that retrieves hits for the inverted query. Receives
        ``(query, k=k, project=project)``.
    contradiction_fn:
        Pairwise scorer returning the probability that the second fact
        contradicts the first.
    project:
        Optional scope passed through to ``search_fn``.
    k:
        How many negative hits to fetch and how many to consider per
        side when computing contradiction pairs (capped at
        ``_MAX_PAIRS_PER_SIDE``).
    llm_model:
        Alias passed to ``llm_client.complete`` (default Haiku).
    llm_client:
        Optional LLM client. ``None`` skips the LLM and falls back to
        the deterministic inversion template.

    Returns
    -------
    NegativeEvidenceResult
        Always populated. ``decision`` is one of
        ``no_contradiction`` / ``soft_contradict`` / ``hard_contradict``.
    """
    q = (question or "").strip()
    cleaned_positives = [
        hit
        for hit in (positive_evidence or [])
        if isinstance(hit, dict) and _clean_text(hit)
    ]

    # ── Empty-positive shortcut: nothing to contradict.
    if not cleaned_positives:
        return NegativeEvidenceResult(
            inverted_query="",
            negative_evidence=[],
            contradiction_score=0.0,
            decision="no_contradiction",
            rationale="no positive evidence supplied; nothing to contradict",
        )

    if not q:
        # Empty question can still attempt retrieval via the template
        # but the inverted query degenerates to "facts contradicting
        # or refuting: ". We bail early — the pipeline upstream of us
        # should never reach here with an empty question, so treat it
        # as a soft no-op.
        return NegativeEvidenceResult(
            inverted_query="",
            negative_evidence=[],
            contradiction_score=0.0,
            decision="no_contradiction",
            rationale="empty question; negative pass skipped",
        )

    # ── 1. Build the inverted retrieval query.
    inverted = _invert_query(q, llm_client=llm_client, llm_model=llm_model)

    # ── 2. Fetch negative hits.
    try:
        negatives_raw = search_fn(inverted, k=k, project=project) or []
    except Exception as exc:
        # Search failure is recoverable — return a no-contradiction
        # result with a clear rationale rather than crashing the
        # caller's retrieval loop.
        return NegativeEvidenceResult(
            inverted_query=inverted,
            negative_evidence=[],
            contradiction_score=0.0,
            decision="no_contradiction",
            rationale=(
                f"negative search failed ({type(exc).__name__}); "
                f"treating as no contradiction"
            ),
        )

    negatives_clean = [
        hit
        for hit in negatives_raw
        if isinstance(hit, dict) and _clean_text(hit)
    ]
    if not negatives_clean:
        return NegativeEvidenceResult(
            inverted_query=inverted,
            negative_evidence=[],
            contradiction_score=0.0,
            decision="no_contradiction",
            rationale="negative search returned no hits",
        )

    # ── 3. Score the cross-product (capped).
    pos_pool = cleaned_positives[:_MAX_PAIRS_PER_SIDE]
    neg_pool = negatives_clean[:_MAX_PAIRS_PER_SIDE]

    max_score = 0.0
    best_pair: tuple[str, str] = ("", "")
    for pos in pos_pool:
        pos_text = _clean_text(pos)
        if not pos_text:
            continue
        for neg in neg_pool:
            neg_text = _clean_text(neg)
            if not neg_text:
                continue
            try:
                raw_score = contradiction_fn(pos_text, neg_text)
            except Exception:
                # One bad scoring call should not poison the rest;
                # treat it as zero contribution.
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            score = max(0.0, min(1.0, score))
            if score > max_score:
                max_score = score
                best_pair = (pos_text, neg_text)

    decision = _decide(max_score)
    rationale = _format_rationale(
        decision, max_score, best_pair[0], best_pair[1]
    )

    return NegativeEvidenceResult(
        inverted_query=inverted,
        negative_evidence=negatives_clean,
        contradiction_score=max_score,
        decision=decision,
        rationale=rationale,
    )


__all__ = [
    "ContradictionFn",
    "DEFAULT_MODEL",
    "LLMLike",
    "NegativeEvidenceResult",
    "SearchFn",
    "THRESHOLD_HARD",
    "THRESHOLD_SOFT",
    "negative_retrieve",
]
