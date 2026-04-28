"""Unit tests for ``memory_core.idk_router`` (W1-D, v11).

Pure-logic module — every test is deterministic, no LLM, no I/O. We
cover all four ``RouteAction`` branches plus boundary conditions on
the thresholds and iteration budget.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory_core.idk_router import (  # noqa: E402
    DEFAULT_THRESHOLD_ANSWER,
    DEFAULT_THRESHOLD_CAVEAT,
    RouteAction,
    RouteDecision,
    route,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


@dataclass
class _Verdict:
    """Lightweight stand-in for ``AnswerabilityResult``.

    The router only reads three fields, so we avoid importing the real
    dataclass and keep the test isolated from the ai_layer.
    """

    answerable: bool = False
    partial: bool = False
    confidence: float = 0.0


def _v(**kw: object) -> _Verdict:
    return _Verdict(**kw)


# ──────────────────────────────────────────────
# Branch 1 — ANSWER
# ──────────────────────────────────────────────


def test_answer_when_answerable_and_above_threshold():
    res = route(_v(answerable=True, partial=False, confidence=0.9))
    assert res.action is RouteAction.ANSWER
    assert res.threshold_used == DEFAULT_THRESHOLD_ANSWER
    assert "answerable" in res.reason


def test_answer_at_exact_threshold():
    res = route(_v(answerable=True, confidence=DEFAULT_THRESHOLD_ANSWER))
    assert res.action is RouteAction.ANSWER


def test_answer_with_custom_higher_threshold():
    # Stricter operator: only answer when confidence >= 0.9.
    res = route(
        _v(answerable=True, confidence=0.85),
        threshold_answer=0.9,
    )
    # 0.85 falls back to the SEARCH/IDK ladder.
    assert res.action is not RouteAction.ANSWER


# ──────────────────────────────────────────────
# Branch 2 — ANSWER_WITH_CAVEAT
# ──────────────────────────────────────────────


def test_caveat_when_partial_and_above_caveat_threshold():
    res = route(_v(partial=True, confidence=0.6))
    assert res.action is RouteAction.ANSWER_WITH_CAVEAT
    assert res.threshold_used == DEFAULT_THRESHOLD_CAVEAT
    assert "partial" in res.reason


def test_caveat_at_exact_caveat_threshold():
    res = route(_v(partial=True, confidence=DEFAULT_THRESHOLD_CAVEAT))
    assert res.action is RouteAction.ANSWER_WITH_CAVEAT


def test_caveat_not_picked_when_partial_but_below_threshold():
    res = route(
        _v(partial=True, confidence=0.30),
        iters_done=4,
        max_iters=4,  # no budget for SEARCH_MORE
    )
    # 0.30 is in uncertain zone (>= 0.225), but no budget → IDK.
    assert res.action is RouteAction.IDK


# ──────────────────────────────────────────────
# Branch 3 — SEARCH_MORE
# ──────────────────────────────────────────────


def test_search_more_in_uncertain_zone_with_budget():
    # caveat_threshold=0.45, lower_bound=0.225 — 0.35 sits inside.
    res = route(
        _v(partial=True, confidence=0.35),
        iters_done=1,
        max_iters=4,
    )
    assert res.action is RouteAction.SEARCH_MORE
    assert "iter 1/4" in res.reason


def test_search_more_when_answerable_but_below_answer_threshold():
    # answerable=True at 0.6 (below 0.75 default) and budget left.
    res = route(
        _v(answerable=True, confidence=0.6),
        iters_done=0,
        max_iters=3,
    )
    assert res.action is RouteAction.SEARCH_MORE


def test_search_more_not_when_budget_exhausted():
    res = route(
        _v(partial=True, confidence=0.35),
        iters_done=4,
        max_iters=4,
    )
    assert res.action is RouteAction.IDK
    assert "budget exhausted" in res.reason


def test_search_more_iters_done_equal_max_iters_blocks_retry():
    res = route(
        _v(partial=True, confidence=0.40),
        iters_done=2,
        max_iters=2,
    )
    assert res.action is RouteAction.IDK


# ──────────────────────────────────────────────
# Branch 4 — IDK
# ──────────────────────────────────────────────


def test_idk_when_no_signals_and_no_budget():
    res = route(
        _v(answerable=False, partial=False, confidence=0.0),
        iters_done=4,
        max_iters=4,
    )
    assert res.action is RouteAction.IDK
    assert res.threshold_used == DEFAULT_THRESHOLD_CAVEAT


def test_idk_below_lower_bound():
    res = route(
        _v(partial=True, confidence=0.10),
        iters_done=0,
        max_iters=4,
    )
    # 0.10 < lower_bound (0.225) → IDK even with budget left.
    assert res.action is RouteAction.IDK


def test_idk_for_negative_confidence_clamped():
    res = route(_v(answerable=False, confidence=-0.5))
    assert res.action is RouteAction.IDK


def test_idk_for_string_confidence_falls_back_to_zero():
    weird = _Verdict(answerable=False, partial=False, confidence="bad")  # type: ignore[arg-type]
    res = route(weird)
    assert res.action is RouteAction.IDK


# ──────────────────────────────────────────────
# Threshold edge cases
# ──────────────────────────────────────────────


def test_custom_thresholds_used_in_decision():
    res = route(
        _v(answerable=True, confidence=0.55),
        threshold_answer=0.5,
        threshold_caveat=0.3,
    )
    assert res.action is RouteAction.ANSWER
    assert res.threshold_used == 0.5


def test_thresholds_clamped_to_unit_range():
    # Out-of-range thresholds must be clamped, not ignored.
    res = route(
        _v(answerable=True, confidence=1.0),
        threshold_answer=1.5,  # clamps to 1.0 → exact-equal hits ANSWER
        threshold_caveat=-0.1,  # clamps to 0.0
    )
    assert res.action is RouteAction.ANSWER


def test_decision_is_immutable_dataclass():
    res = route(_v(answerable=True, confidence=0.99))
    assert isinstance(res, RouteDecision)
    assert isinstance(res.action, RouteAction)
    assert isinstance(res.action.value, str)


# ──────────────────────────────────────────────
# Defensive — duck-typed verdict
# ──────────────────────────────────────────────


def test_route_accepts_arbitrary_object_with_required_fields():
    class _AdHoc:
        answerable = True
        partial = False
        confidence = 0.8

    res = route(_AdHoc())
    assert res.action is RouteAction.ANSWER


def test_route_handles_missing_attributes_safely():
    class _Sparse:
        pass

    # No attrs → defaults to False/0.0, no budget → IDK.
    res = route(_Sparse(), iters_done=1, max_iters=1)
    assert res.action is RouteAction.IDK
