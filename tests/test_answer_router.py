"""Unit tests for ``memory_core.answer_router`` (W2-I, v11).

Pure-logic module — no LLM, no I/O. We cover every branch of the
decision tree explicitly and make sure the router stays
**permissive**: per failure-analysis 30% of errors today are
``over_cautious``, so the router must not abstain when evidence is
present and the score barely clears the threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory_core.answer_router import (  # noqa: E402
    DEFAULT_PER_CATEGORY_THRESHOLDS,
    RouteAction,
    RoutingDecision,
    RoutingInputs,
    route,
)
from memory_core.calibration import PlattCalibrator  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _inputs(**overrides) -> RoutingInputs:
    """Build a ``RoutingInputs`` with sensible defaults overridable per test."""
    base = dict(
        category="single-hop",
        raw_retrieval_score=0.5,
        answerable=True,
        partial_answerable=False,
        answerability_confidence=0.7,
        nli_decision="entail",
        nli_p_contradict=0.05,
        iters_done=0,
        max_iters=3,
        has_contradiction=False,
    )
    base.update(overrides)
    return RoutingInputs(**base)


# Identity-ish calibrator: p(s) = sigmoid(12s - 6).
# Mapping: s=0.0 → p≈0.0025, s=0.5 → p=0.5, s=1.0 → p≈0.998.
# Calibrator parametrisation: p = 1/(1+exp(a*s+b)), so we need
# a*s+b = -(12s-6) = -12s+6 → a=-12, b=6.
def _identity_calibrator() -> PlattCalibrator:
    return PlattCalibrator(a=-12.0, b=6.0)


# ────────────────────────────────────────────────────────────────────
# Branch 1: NLI verifier veto
# ────────────────────────────────────────────────────────────────────


def test_verifier_veto_fires_on_strong_contradict() -> None:
    """nli_decision=contradict and p_contradict>=0.7 → IDK."""
    decision = route(
        _inputs(
            answerable=True,
            raw_retrieval_score=0.95,
            nli_decision="contradict",
            nli_p_contradict=0.85,
        )
    )
    assert decision.action is RouteAction.IDK
    assert "verifier veto" in decision.reason.lower()


def test_verifier_veto_skipped_on_weak_contradict() -> None:
    """contradict but p<0.7 → fall through to normal routing."""
    decision = route(
        _inputs(
            answerable=True,
            raw_retrieval_score=0.95,
            nli_decision="contradict",
            nli_p_contradict=0.3,
        )
    )
    assert decision.action is RouteAction.ANSWER


# ────────────────────────────────────────────────────────────────────
# Branch 2: negative-retrieval contradiction veto
# ────────────────────────────────────────────────────────────────────


def test_contradiction_flag_forces_idk() -> None:
    """has_contradiction=True → IDK regardless of score."""
    decision = route(
        _inputs(
            answerable=True,
            raw_retrieval_score=0.99,
            has_contradiction=True,
        )
    )
    assert decision.action is RouteAction.IDK
    assert "contradiction" in decision.reason.lower()


# ────────────────────────────────────────────────────────────────────
# Branch 5: ANSWER (above category threshold)
# ────────────────────────────────────────────────────────────────────


def test_answer_when_above_threshold_single_hop() -> None:
    """Single-hop threshold is 0.4; raw_score 0.7 → ANSWER."""
    decision = route(_inputs(category="single-hop", raw_retrieval_score=0.7))
    assert decision.action is RouteAction.ANSWER
    assert decision.threshold_used == pytest.approx(0.40)


def test_answer_when_calibrated_just_above_threshold() -> None:
    """PERMISSIVE check: p just above threshold must yield ANSWER, not IDK.

    With the identity-ish calibrator p = sigmoid(12s - 6), raw_score=0.55
    gives p ≈ sigmoid(0.6) ≈ 0.646 — clearly above the single-hop
    threshold of 0.40. The router must NOT fall through to IDK here.
    """
    decision = route(
        _inputs(
            category="single-hop",
            raw_retrieval_score=0.55,
            answerable=True,
        ),
        calibrator=_identity_calibrator(),
    )
    assert decision.action is RouteAction.ANSWER, (
        f"PERMISSIVE: expected ANSWER, got {decision.action} "
        f"(p={decision.calibrated_p:.3f}, reason={decision.reason})"
    )
    assert decision.calibrated_p > 0.40


def test_answer_uses_calibrator_when_provided() -> None:
    """With calibrator, the decision is driven by calibrated p, not raw."""
    decision = route(
        _inputs(category="adversarial", raw_retrieval_score=0.95),
        calibrator=_identity_calibrator(),
    )
    # adv threshold 0.7, calibrated p≈0.999 — clear ANSWER.
    assert decision.action is RouteAction.ANSWER
    assert decision.calibrated_p > 0.9


# ────────────────────────────────────────────────────────────────────
# Branch 6: ANSWER_WITH_CAVEAT
# ────────────────────────────────────────────────────────────────────


def test_caveat_when_partial_evidence_and_score_in_caveat_band() -> None:
    """answerable=False, partial=True, p in [threshold*0.7, threshold)."""
    # multi-hop threshold 0.5, caveat floor 0.35. p≈0.4.
    decision = route(
        _inputs(
            category="multi-hop",
            raw_retrieval_score=0.4,
            answerable=False,
            partial_answerable=True,
        )
    )
    assert decision.action is RouteAction.ANSWER_WITH_CAVEAT
    assert decision.threshold_used == pytest.approx(0.5 * 0.7)


def test_caveat_does_not_fire_when_score_too_low() -> None:
    """partial=True but p below caveat floor → SEARCH_MORE (budget) or IDK."""
    decision = route(
        _inputs(
            category="multi-hop",
            raw_retrieval_score=0.1,
            answerable=False,
            partial_answerable=True,
            iters_done=0,
            max_iters=3,
        )
    )
    assert decision.action is RouteAction.SEARCH_MORE


# ────────────────────────────────────────────────────────────────────
# Branch 7: HYBRID (open-domain mid-band)
# ────────────────────────────────────────────────────────────────────


def test_hybrid_for_open_domain_mid_band() -> None:
    """open-domain, answerable=True, p ∈ [threshold*0.5, threshold) → HYBRID.

    open threshold = 0.40, hybrid floor = 0.20. raw_score 0.30 sits in
    the mid-band when the calibrator is not provided (raw used as p).
    """
    decision = route(
        _inputs(
            category="open-domain",
            raw_retrieval_score=0.30,
            answerable=True,
            partial_answerable=False,
        )
    )
    assert decision.action is RouteAction.HYBRID
    assert decision.threshold_used == pytest.approx(0.20)


def test_hybrid_does_not_fire_outside_open_domain() -> None:
    """Same mid-band score on single-hop should NOT yield HYBRID."""
    decision = route(
        _inputs(
            category="single-hop",
            raw_retrieval_score=0.22,
            answerable=True,
            partial_answerable=False,
            iters_done=3,
            max_iters=3,  # no budget — forces non-SEARCH branch
        )
    )
    assert decision.action is not RouteAction.HYBRID


def test_hybrid_requires_answerable_true() -> None:
    """HYBRID needs answerable=True; otherwise we either search or IDK."""
    decision = route(
        _inputs(
            category="open-domain",
            raw_retrieval_score=0.30,
            answerable=False,
            partial_answerable=False,
            iters_done=3,
            max_iters=3,
        )
    )
    assert decision.action is not RouteAction.HYBRID


# ────────────────────────────────────────────────────────────────────
# Branch 8: SEARCH_MORE — low confidence with budget
# ────────────────────────────────────────────────────────────────────


def test_search_more_when_low_score_and_budget() -> None:
    """p below hybrid floor and budget remains → SEARCH_MORE."""
    decision = route(
        _inputs(
            category="single-hop",
            raw_retrieval_score=0.05,
            answerable=False,
            partial_answerable=False,
            iters_done=1,
            max_iters=3,
        )
    )
    assert decision.action is RouteAction.SEARCH_MORE


# ────────────────────────────────────────────────────────────────────
# Branch 9: SEARCH_MORE — not answerable but budget remains
# ────────────────────────────────────────────────────────────────────


def test_search_more_when_not_answerable_with_budget() -> None:
    """answerable=False, score in mid-band, budget remains → SEARCH_MORE.

    Score 0.25 on multi-hop: hybrid_floor=0.25, threshold=0.5.
    Not open-domain so HYBRID doesn't fire. partial=False so caveat
    doesn't fire. Budget remains and answerable=False → SEARCH_MORE.
    """
    decision = route(
        _inputs(
            category="multi-hop",
            raw_retrieval_score=0.27,
            answerable=False,
            partial_answerable=False,
            iters_done=1,
            max_iters=3,
        )
    )
    assert decision.action is RouteAction.SEARCH_MORE


# ────────────────────────────────────────────────────────────────────
# Branch 10: IDK as last resort
# ────────────────────────────────────────────────────────────────────


def test_idk_as_last_resort_no_budget_low_score() -> None:
    """No budget, not answerable, no partial → IDK (last resort)."""
    decision = route(
        _inputs(
            category="single-hop",
            raw_retrieval_score=0.05,
            answerable=False,
            partial_answerable=False,
            iters_done=3,
            max_iters=3,
        )
    )
    assert decision.action is RouteAction.IDK


def test_idk_when_no_budget_and_score_below_threshold() -> None:
    """Budget exhausted, answerable=True but p<threshold and not in HYBRID band."""
    decision = route(
        _inputs(
            category="adversarial",
            raw_retrieval_score=0.30,  # < 0.7 adv threshold, > 0.35 hybrid floor of 0.35
            answerable=True,
            partial_answerable=False,
            iters_done=3,
            max_iters=3,
        )
    )
    # Not open-domain so no HYBRID; budget exhausted so no SEARCH_MORE.
    # answerable but p<threshold so no ANSWER. Falls through to IDK.
    assert decision.action is RouteAction.IDK


# ────────────────────────────────────────────────────────────────────
# Per-category threshold influence
# ────────────────────────────────────────────────────────────────────


def test_same_score_different_category_different_action() -> None:
    """Same score 0.45 acts as ANSWER for single-hop, IDK/SEARCH for adversarial.

    single-hop threshold 0.40 → ANSWER.
    adversarial threshold 0.70 → not ANSWER. With budget, SEARCH_MORE.
    """
    score = 0.45
    single = route(_inputs(category="single-hop", raw_retrieval_score=score))
    assert single.action is RouteAction.ANSWER

    adv = route(
        _inputs(
            category="adversarial",
            raw_retrieval_score=score,
            iters_done=0,
            max_iters=3,
        )
    )
    assert adv.action is RouteAction.SEARCH_MORE


def test_unknown_category_falls_back_to_default() -> None:
    """Categories not in the table use 'default' = 0.45."""
    decision = route(
        _inputs(category="weird-new-category", raw_retrieval_score=0.5)
    )
    assert decision.action is RouteAction.ANSWER
    assert decision.threshold_used == pytest.approx(0.45)


def test_category_none_uses_default_threshold() -> None:
    """category=None resolves to the default bucket."""
    decision = route(_inputs(category=None, raw_retrieval_score=0.6))
    assert decision.action is RouteAction.ANSWER
    assert decision.threshold_used == pytest.approx(
        DEFAULT_PER_CATEGORY_THRESHOLDS["default"]
    )


def test_custom_thresholds_override_defaults() -> None:
    """Caller-provided table overrides DEFAULT_PER_CATEGORY_THRESHOLDS."""
    custom = {"single-hop": 0.95, "default": 0.95}
    decision = route(
        _inputs(category="single-hop", raw_retrieval_score=0.5),
        per_category_thresholds=custom,
    )
    # 0.5 < 0.95 and budget remains, score above hybrid floor (0.475)
    # but not open-domain → SEARCH_MORE branch.
    assert decision.action is RouteAction.SEARCH_MORE


def test_decision_echoes_threshold_table() -> None:
    """The decision returns a copy of the threshold table consulted."""
    decision = route(_inputs(category="single-hop", raw_retrieval_score=0.7))
    assert "single-hop" in decision.per_category_threshold
    assert decision.per_category_threshold["single-hop"] == pytest.approx(0.40)


# ────────────────────────────────────────────────────────────────────
# Defensive parsing
# ────────────────────────────────────────────────────────────────────


def test_garbage_nli_p_contradict_does_not_crash() -> None:
    """Non-finite p_contradict is treated as 0 — no veto, no exception."""
    decision = route(
        _inputs(
            answerable=True,
            raw_retrieval_score=0.7,
            nli_decision="contradict",
            nli_p_contradict=float("nan"),
        )
    )
    # NaN coerces to 0 → veto does not fire → ANSWER branch is reached.
    assert decision.action is RouteAction.ANSWER


def test_negative_iters_clamped_to_zero() -> None:
    """iters_done < 0 is clamped — does not enable infinite SEARCH_MORE."""
    decision = route(
        _inputs(
            category="single-hop",
            raw_retrieval_score=0.05,
            answerable=False,
            partial_answerable=False,
            iters_done=-5,
            max_iters=2,
        )
    )
    # iters_done clamped to 0 → budget remains → SEARCH_MORE.
    assert decision.action is RouteAction.SEARCH_MORE


# ────────────────────────────────────────────────────────────────────
# Decision invariants
# ────────────────────────────────────────────────────────────────────


def test_calibrated_p_is_in_unit_interval_for_all_branches() -> None:
    """Across all sample inputs the reported calibrated_p stays in [0, 1]."""
    samples = [
        _inputs(raw_retrieval_score=-1.0),
        _inputs(raw_retrieval_score=0.0),
        _inputs(raw_retrieval_score=0.5),
        _inputs(raw_retrieval_score=1.0),
        _inputs(raw_retrieval_score=10.0),
    ]
    for s in samples:
        d = route(s, calibrator=_identity_calibrator())
        assert 0.0 <= d.calibrated_p <= 1.0, (
            f"calibrated_p={d.calibrated_p} for inputs {s}"
        )


def test_decision_is_immutable() -> None:
    """``RoutingDecision`` is frozen — accidental mutation raises."""
    decision = route(_inputs())
    with pytest.raises(Exception):
        decision.action = RouteAction.IDK  # type: ignore[misc]
