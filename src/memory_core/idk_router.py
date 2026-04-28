"""IDK router (W1-D, v11) — turn an answerability verdict into a route.

This is the deterministic counterpart to
``ai_layer.answerability.classify_answerability``. The classifier sees
the LLM; this router never does. Given a verdict and the iteration
budget the recall loop has left, it picks one of four actions:

* ``ANSWER``              — fully grounded, generator can write.
* ``ANSWER_WITH_CAVEAT``  — partial evidence, write with hedging.
* ``SEARCH_MORE``         — confidence in the gap zone and budget left.
* ``IDK``                 — not answerable; return "I don't know".

Living in ``memory_core`` is intentional: routing is pure logic on
small dataclasses, has no LLM dependency, and runs on the hot path of
every retrieval iteration. The v11 layer-separation regression test
(``tests/test_v11_layer_separation.py``) forbids ``memory_core``
modules from importing ``ai_layer`` — so we import only the dataclass
through a TYPE_CHECKING-guarded reference for typing, and accept any
duck-typed object at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class _AnswerabilityLike(Protocol):
    """Subset of ``AnswerabilityResult`` the router actually reads.

    Defined as a structural type so ``memory_core`` does not need to
    import ``ai_layer`` at runtime — the layer wall is enforced by
    ``test_v11_layer_separation``.
    """

    answerable: bool
    partial: bool
    confidence: float


# ──────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────


class RouteAction(str, Enum):
    """Possible outputs of :func:`route`.

    Inheriting from ``str`` keeps JSON-serialisability and lets callers
    compare directly against literal action names from configs.
    """

    ANSWER = "answer"
    ANSWER_WITH_CAVEAT = "answer_with_caveat"
    SEARCH_MORE = "search_more"
    IDK = "idk"


@dataclass
class RouteDecision:
    """The router's verdict for one iteration of the recall loop."""

    action: RouteAction
    threshold_used: float
    reason: str


# ──────────────────────────────────────────────
# Default thresholds (kept in one place for tuning)
# ──────────────────────────────────────────────

# Minimum confidence to ANSWER outright. Below this we hedge or search.
DEFAULT_THRESHOLD_ANSWER = 0.75
# Minimum confidence to even attempt a hedged answer.
DEFAULT_THRESHOLD_CAVEAT = 0.45
# Lower bound below which we stop spending iterations and bail to IDK.
# Picked at half of the caveat threshold — anything weaker than that
# is signal noise, not a partial answer worth searching for.
_LOWER_BOUND_FACTOR = 0.5


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────


def route(
    answerability: _AnswerabilityLike,
    *,
    iters_done: int = 0,
    max_iters: int = 4,
    threshold_answer: float = DEFAULT_THRESHOLD_ANSWER,
    threshold_caveat: float = DEFAULT_THRESHOLD_CAVEAT,
) -> RouteDecision:
    """Route an answerability verdict to a concrete next step.

    Parameters
    ----------
    answerability:
        Output of ``classify_answerability``. Any object with the same
        fields works — see :class:`_AnswerabilityLike`.
    iters_done:
        How many retrieval rounds have already run. ``0`` = first.
    max_iters:
        Hard cap on retrieval rounds. ``iters_done >= max_iters``
        forbids further ``SEARCH_MORE`` decisions.
    threshold_answer / threshold_caveat:
        Confidence thresholds for ANSWER and ANSWER_WITH_CAVEAT.

    Returns
    -------
    RouteDecision
        ``threshold_used`` is the threshold that drove the decision —
        useful for logging and bench analysis.

    Logic
    -----
    1. ``answerable=True`` and confidence >= ``threshold_answer``
       → ``ANSWER``.
    2. ``partial=True`` and confidence >= ``threshold_caveat``
       → ``ANSWER_WITH_CAVEAT``.
    3. There is budget left and confidence sits in the "uncertain zone"
       (between the lower bound and the caveat threshold)
       → ``SEARCH_MORE``.
    4. Anything else → ``IDK``.

    Edge cases
    ----------
    * Negative or out-of-range confidence is clamped to ``[0.0, 1.0]``.
    * ``threshold_answer < threshold_caveat`` is allowed but the result
      collapses to a single threshold — we never silently re-order
      caller config.
    """
    answerable = bool(getattr(answerability, "answerable", False))
    partial = bool(getattr(answerability, "partial", False))
    raw_conf = getattr(answerability, "confidence", 0.0)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    threshold_answer = max(0.0, min(1.0, float(threshold_answer)))
    threshold_caveat = max(0.0, min(1.0, float(threshold_caveat)))
    lower_bound = threshold_caveat * _LOWER_BOUND_FACTOR
    iters_done = max(0, int(iters_done))
    max_iters = max(0, int(max_iters))
    has_budget = iters_done < max_iters

    # ── 1. Strong, fully-grounded answer.
    if answerable and confidence >= threshold_answer:
        return RouteDecision(
            action=RouteAction.ANSWER,
            threshold_used=threshold_answer,
            reason=(
                f"answerable with confidence {confidence:.2f} "
                f">= {threshold_answer:.2f}"
            ),
        )

    # ── 2. Partial — hedge if confidence supports it.
    if partial and confidence >= threshold_caveat:
        return RouteDecision(
            action=RouteAction.ANSWER_WITH_CAVEAT,
            threshold_used=threshold_caveat,
            reason=(
                f"partial evidence at confidence {confidence:.2f} "
                f">= caveat threshold {threshold_caveat:.2f}"
            ),
        )

    # ── 3. Uncertain zone — keep searching while we have budget.
    in_uncertain_zone = lower_bound <= confidence < threshold_caveat
    # Also retry when the model said "answerable" but did not clear the
    # answer threshold — extra context might tip it over.
    near_answer_threshold = (
        answerable and lower_bound <= confidence < threshold_answer
    )
    if has_budget and (in_uncertain_zone or near_answer_threshold):
        return RouteDecision(
            action=RouteAction.SEARCH_MORE,
            threshold_used=threshold_caveat,
            reason=(
                f"confidence {confidence:.2f} in uncertain zone "
                f"[{lower_bound:.2f}, {threshold_caveat:.2f}); "
                f"iter {iters_done}/{max_iters}, retrying"
            ),
        )

    # ── 4. Fall through — give up cleanly.
    if not has_budget:
        budget_note = f"iteration budget exhausted ({iters_done}/{max_iters})"
    else:
        budget_note = (
            f"confidence {confidence:.2f} below lower bound {lower_bound:.2f}"
        )
    return RouteDecision(
        action=RouteAction.IDK,
        threshold_used=threshold_caveat,
        reason=(
            f"insufficient evidence — {budget_note}; "
            f"answerable={answerable}, partial={partial}"
        ),
    )


__all__ = [
    "DEFAULT_THRESHOLD_ANSWER",
    "DEFAULT_THRESHOLD_CAVEAT",
    "RouteAction",
    "RouteDecision",
    "route",
]
