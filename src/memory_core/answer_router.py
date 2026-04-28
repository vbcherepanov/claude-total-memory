"""Answer router (W2-I, v11) — calibrated routing on top of ``idk_router``.

Why this module
---------------
``idk_router.route`` uses raw answerability confidence. After W2-I we
have two new signals:

* a fitted :class:`PlattCalibrator` mapping raw retrieval scores to
  calibrated p(correct);
* an NLI verifier verdict (``entail``/``neutral``/``contradict``) plus
  a contradiction probability.

We combine them here. The routing is deliberately **permissive** —
LoCoMo failure analysis (``benchmarks/results/baseline-failure-analysis.md``)
shows 30% of errors are "over_cautious", i.e. abstaining when the
evidence does support an answer. The router must not make that worse.

Layer separation
----------------
``memory_core`` cannot import ``ai_layer``. We therefore type the NLI
decision as ``str`` and accept it from the caller. This keeps the layer
wall green while still giving the router everything it needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from memory_core.calibration import PlattCalibrator, apply


__all__ = [
    "DEFAULT_PER_CATEGORY_THRESHOLDS",
    "RouteAction",
    "RoutingDecision",
    "RoutingInputs",
    "route",
]


# ────────────────────────────────────────────────────────────────────
# Public types
# ────────────────────────────────────────────────────────────────────


class RouteAction(str, Enum):
    """Possible terminal actions from the router.

    Distinct from ``idk_router.RouteAction``: this layer adds
    ``HYBRID`` (memory + LLM parametric knowledge for open-domain Qs).
    """

    ANSWER = "answer"
    ANSWER_WITH_CAVEAT = "answer_with_caveat"
    HYBRID = "hybrid"
    SEARCH_MORE = "search_more"
    IDK = "idk"


@dataclass(frozen=True)
class RoutingInputs:
    """Everything ``route`` needs to make a decision.

    Frozen so a single ``RoutingInputs`` can be reused across log lines
    without surprise mutation. ``category`` is optional because we may
    not have a classifier verdict yet on a given iteration.
    """

    category: str | None
    raw_retrieval_score: float
    answerable: bool
    partial_answerable: bool
    answerability_confidence: float
    nli_decision: str  # 'entail' | 'neutral' | 'contradict'
    nli_p_contradict: float
    iters_done: int
    max_iters: int
    has_contradiction: bool


@dataclass(frozen=True)
class RoutingDecision:
    """The router's verdict for one recall-loop iteration."""

    action: RouteAction
    calibrated_p: float
    threshold_used: float
    reason: str
    per_category_threshold: Mapping[str, float] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Defaults
# ────────────────────────────────────────────────────────────────────


# Per-category thresholds derived from the v9 archive analysis
# (baseline-failure-analysis.md). The numbers are biased toward
# answering rather than abstaining; adversarial sits high to avoid
# hallucinations on questions with no answer in memory.
#
# Keys deliberately use both the LoCoMo-style spellings ("multi-hop")
# and the abbreviated forms used in the failure-analysis table. The
# lookup helper canonicalises before reading.
DEFAULT_PER_CATEGORY_THRESHOLDS: dict[str, float] = {
    "single-hop": 0.40,
    "single": 0.40,
    "multi-hop": 0.50,
    "multi": 0.50,
    "temporal": 0.45,
    "temp": 0.45,
    "open-domain": 0.40,
    "open": 0.40,
    "adversarial": 0.70,
    "adv": 0.70,
    "default": 0.45,
}


# Minimum p(contradict) above which the verifier veto fires.
_VERIFIER_VETO_PCONTRA = 0.7

# Multipliers used by the routing rules. Kept named so the rule
# numbers in docstrings stay readable.
_CAVEAT_FACTOR = 0.7
_HYBRID_FLOOR_FACTOR = 0.5  # also: SEARCH_MORE-when-low boundary


def _category_threshold(
    category: str | None, table: Mapping[str, float]
) -> tuple[str, float]:
    """Return (key_used, threshold) for ``category``.

    Falls back to ``default`` when the category is None or absent. We
    return the key actually consulted for transparency in the decision
    ``reason`` field.
    """
    if category is None:
        key = "default"
    else:
        key = category if category in table else "default"
    threshold = float(table.get(key, table.get("default", 0.45)))
    return key, max(0.0, min(1.0, threshold))


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


# ────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────


def route(
    inputs: RoutingInputs,
    *,
    calibrator: PlattCalibrator | None = None,
    per_category_thresholds: Mapping[str, float] | None = None,
) -> RoutingDecision:
    """Pick a :class:`RouteAction` for the current iteration.

    The decision tree is intentionally **permissive** — see the module
    docstring. Order of checks:

    1. **Verifier veto.** ``nli_decision == 'contradict'`` *and*
       ``nli_p_contradict >= 0.7`` → IDK. The verifier outranks
       retrieval confidence: if NLI says the candidate answer is
       contradicted by evidence we abstain regardless of score.
    2. **Negative-retrieval veto.** ``has_contradiction`` flag from the
       negative_retrieve pipeline → IDK.
    3. **Calibrate.** ``p = apply(calibrator, raw_score)`` if a
       calibrator was provided, otherwise ``p = clamp(raw_score)``.
    4. **Lookup threshold** for the category.
    5. ``answerable`` and ``p >= threshold`` → ANSWER.
    6. ``partial_answerable`` and ``p >= threshold * 0.7`` →
       ANSWER_WITH_CAVEAT.
    7. **HYBRID** when ``category == 'open-domain'`` and
       ``answerable`` and ``p`` sits in
       ``[threshold*0.5, threshold)``. Use memory + LLM parametric
       knowledge with a caveat. Checked **before** SEARCH_MORE so the
       router never wastes budget on open-domain questions where the
       LLM can step in.
    8. ``iters_done < max_iters`` and ``p < threshold * 0.5`` →
       SEARCH_MORE (low confidence but budget remains).
    9. ``iters_done < max_iters`` and not answerable → SEARCH_MORE.
    10. else → IDK (last resort).

    Notes
    -----
    * The calibrator is optional so the router degrades to "use raw
      score as probability" when none is loaded — important for
      bootstrap and tests.
    * ``per_category_threshold`` is echoed back in the decision so
      logging downstream can show *which* table was consulted; we
      always pass a copy to avoid leaking caller's dict mutation.
    """
    table = (
        dict(per_category_thresholds)
        if per_category_thresholds is not None
        else dict(DEFAULT_PER_CATEGORY_THRESHOLDS)
    )

    # 1. Verifier veto.
    p_contra = _safe_float(inputs.nli_p_contradict)
    if (
        inputs.nli_decision == "contradict"
        and p_contra >= _VERIFIER_VETO_PCONTRA
    ):
        return RoutingDecision(
            action=RouteAction.IDK,
            calibrated_p=0.0,
            threshold_used=_VERIFIER_VETO_PCONTRA,
            reason=(
                f"NLI verifier veto: contradict at p_contradict="
                f"{p_contra:.2f} >= {_VERIFIER_VETO_PCONTRA:.2f}"
            ),
            per_category_threshold=table,
        )

    # 2. Negative-retrieval veto.
    if inputs.has_contradiction:
        return RoutingDecision(
            action=RouteAction.IDK,
            calibrated_p=0.0,
            threshold_used=0.0,
            reason="negative_retrieve flagged a contradiction in evidence",
            per_category_threshold=table,
        )

    # 3. Calibrate (or pass-through clamp).
    raw = _safe_float(inputs.raw_retrieval_score)
    if calibrator is not None:
        p = apply(calibrator, raw)
    else:
        p = max(0.0, min(1.0, raw))

    # 4. Threshold lookup.
    key_used, threshold = _category_threshold(inputs.category, table)
    caveat_floor = threshold * _CAVEAT_FACTOR
    hybrid_floor = threshold * _HYBRID_FLOOR_FACTOR

    iters_done = max(0, int(inputs.iters_done))
    max_iters = max(0, int(inputs.max_iters))
    has_budget = iters_done < max_iters

    # 5. Strong answer.
    if inputs.answerable and p >= threshold:
        return RoutingDecision(
            action=RouteAction.ANSWER,
            calibrated_p=p,
            threshold_used=threshold,
            reason=(
                f"answerable and calibrated p={p:.3f} >= "
                f"category[{key_used}] threshold={threshold:.3f}"
            ),
            per_category_threshold=table,
        )

    # 6. Hedged answer on partial evidence.
    if inputs.partial_answerable and p >= caveat_floor:
        return RoutingDecision(
            action=RouteAction.ANSWER_WITH_CAVEAT,
            calibrated_p=p,
            threshold_used=caveat_floor,
            reason=(
                f"partial evidence and p={p:.3f} >= caveat floor "
                f"{caveat_floor:.3f} (= {threshold:.3f} * "
                f"{_CAVEAT_FACTOR})"
            ),
            per_category_threshold=table,
        )

    # 7. Hybrid — open-domain only, mid-band confidence with answerable.
    if (
        inputs.category in ("open-domain", "open")
        and inputs.answerable
        and hybrid_floor <= p < threshold
    ):
        return RoutingDecision(
            action=RouteAction.HYBRID,
            calibrated_p=p,
            threshold_used=hybrid_floor,
            reason=(
                f"open-domain mid-band: p={p:.3f} in "
                f"[{hybrid_floor:.3f}, {threshold:.3f}) — combine "
                f"memory with LLM parametric knowledge"
            ),
            per_category_threshold=table,
        )

    # 8. Low confidence with budget — search more.
    if has_budget and p < hybrid_floor:
        return RoutingDecision(
            action=RouteAction.SEARCH_MORE,
            calibrated_p=p,
            threshold_used=hybrid_floor,
            reason=(
                f"p={p:.3f} below hybrid floor {hybrid_floor:.3f} "
                f"and budget remains (iter {iters_done}/{max_iters})"
            ),
            per_category_threshold=table,
        )

    # 9. Not answerable but budget remains — give retrieval another go.
    if has_budget and not inputs.answerable:
        return RoutingDecision(
            action=RouteAction.SEARCH_MORE,
            calibrated_p=p,
            threshold_used=threshold,
            reason=(
                f"not answerable yet and budget remains "
                f"(iter {iters_done}/{max_iters}); searching more"
            ),
            per_category_threshold=table,
        )

    # 9b. Permissive fallback (W2-I): when budget remains and we did NOT
    # match ANSWER / CAVEAT / HYBRID, prefer SEARCH_MORE over IDK. The
    # failure analysis (baseline-failure-analysis.md) shows 30% of
    # baseline errors are "over_cautious" — defaulting to IDK whenever
    # we still have a retrieval round in the budget would *increase*
    # that. Only when iterations are exhausted do we fall through to
    # the last-resort IDK below.
    if has_budget:
        return RoutingDecision(
            action=RouteAction.SEARCH_MORE,
            calibrated_p=p,
            threshold_used=threshold,
            reason=(
                f"permissive: p={p:.3f} below threshold={threshold:.3f} "
                f"and no strong branch fired; iter {iters_done}/{max_iters} — "
                f"trying another retrieval round"
            ),
            per_category_threshold=table,
        )

    # 10. Last resort.
    return RoutingDecision(
        action=RouteAction.IDK,
        calibrated_p=p,
        threshold_used=threshold,
        reason=(
            f"no branch matched: p={p:.3f}, threshold={threshold:.3f}, "
            f"answerable={inputs.answerable}, "
            f"partial={inputs.partial_answerable}, "
            f"iter {iters_done}/{max_iters}"
        ),
        per_category_threshold=table,
    )
