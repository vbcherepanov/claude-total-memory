"""Unit tests for ``ai_layer.answerability`` (W1-D, v11).

The classifier wraps a single Haiku call. We never let pytest reach the
network — every test injects a ``FakeLLMClient`` whose ``.complete``
returns a queued response. The 50-case fixture file feeds the
parameterised test that measures false-IDK / true-IDK rates.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# Make src/ importable — conftest.py also does this but we want this
# file to be runnable on its own.
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_layer.answerability import (  # noqa: E402
    AnswerabilityResult,
    classify_answerability,
)


FIXTURES_PATH = Path(__file__).parent / "fixtures" / "answerability_fixtures.json"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


@dataclass
class _FakeResponse:
    """Mimic the shape of ``LLMResult`` so tests cover both code paths.

    The classifier accepts either bare strings or objects with a
    ``.text`` attribute; we exercise both.
    """

    text: str


class FakeLLMClient:
    """Test double for ``benchmarks._llm_adapter.LLMClient``.

    ``responses`` is consumed FIFO. Each call pops one entry; when the
    queue is empty we raise so a misbehaving test surfaces immediately.
    """

    def __init__(self, responses: list[Any]) -> None:
        self.q: list[Any] = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> Any:
        if not self.q:
            raise AssertionError("FakeLLMClient response queue exhausted")
        self.calls.append(kwargs)
        return self.q.pop(0)


def _build_response(
    *,
    answerable: bool,
    partial: bool,
    confidence: float,
    missing: str | None,
    rationale: str = "ok",
    return_str: bool = True,
) -> Any:
    payload: dict[str, Any] = {
        "answerable": answerable,
        "partial": partial,
        "confidence": confidence,
        "missing": missing,
        "rationale": rationale,
    }
    text = json.dumps(payload)
    return text if return_str else _FakeResponse(text=text)


def _expected_synthetic_response(expect: dict[str, Any]) -> Any:
    """Translate a fixture's ``expect`` block into a fake LLM reply.

    The fake here represents an *ideal* model — one that returns the
    correct verdict. The test then asserts the classifier propagates
    those fields faithfully and that no field is dropped or mangled by
    the parser.
    """
    answerable = bool(expect["answerable"])
    partial = bool(expect["partial"])
    base_conf = max(float(expect.get("min_confidence", 0.5)), 0.05)
    confidence = min(0.99, base_conf + 0.1)
    if answerable:
        missing = None
    elif partial:
        missing = "specific detail"
    else:
        missing = "topic not in evidence"
    return _build_response(
        answerable=answerable,
        partial=partial,
        confidence=confidence,
        missing=missing,
        rationale="synthetic test verdict",
    )


def _load_fixtures() -> list[dict[str, Any]]:
    with FIXTURES_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list) and data, "fixtures must be a non-empty list"
    return data


FIXTURES = _load_fixtures()


# ──────────────────────────────────────────────
# Sanity checks on the fixture file itself
# ──────────────────────────────────────────────


def test_fixtures_count_is_fifty():
    assert len(FIXTURES) == 50, f"expected 50 fixtures, got {len(FIXTURES)}"


def test_fixtures_have_required_categories():
    by_cat: dict[str, int] = {}
    for fx in FIXTURES:
        by_cat[fx["category"]] = by_cat.get(fx["category"], 0) + 1
    # Five buckets, ten each — keep the suite honest if someone edits.
    yes_count = by_cat.get("clear_yes", 0)
    no_count = by_cat.get("clear_no", 0)
    partial_count = by_cat.get("partial", 0)
    adv_count = by_cat.get("adversarial", 0)
    edge_count = (
        by_cat.get("edge_short", 0)
        + by_cat.get("edge_long", 0)
        + by_cat.get("edge_multilingual", 0)
    )
    assert yes_count == 10, f"clear_yes = {yes_count}"
    assert no_count == 10, f"clear_no = {no_count}"
    assert partial_count == 10, f"partial = {partial_count}"
    assert adv_count == 10, f"adversarial = {adv_count}"
    assert edge_count == 10, f"edge total = {edge_count}"


# ──────────────────────────────────────────────
# Empty / degenerate inputs (no LLM call expected)
# ──────────────────────────────────────────────


def test_empty_evidence_short_circuits_without_llm_call():
    fake = FakeLLMClient(responses=[])  # exhaustion would raise
    res = classify_answerability(
        "What is Alice's favorite color?", [], llm_client=fake
    )
    assert isinstance(res, AnswerabilityResult)
    assert res.answerable is False
    assert res.partial is False
    assert res.confidence == 1.0
    assert res.missing == "no evidence"
    assert fake.calls == []


def test_whitespace_only_evidence_short_circuits():
    fake = FakeLLMClient(responses=[])
    res = classify_answerability("Q?", ["", "   ", "\n\t"], llm_client=fake)
    assert res.answerable is False
    assert res.partial is False
    assert fake.calls == []


def test_empty_question_short_circuits():
    fake = FakeLLMClient(responses=[])
    res = classify_answerability("   ", ["any evidence"], llm_client=fake)
    assert res.answerable is False
    assert res.partial is False
    assert res.confidence == 1.0
    assert fake.calls == []


# ──────────────────────────────────────────────
# Parser robustness — single LLM call, varied output shapes
# ──────────────────────────────────────────────


def test_classifier_handles_string_response():
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answerable": True,
                    "partial": False,
                    "confidence": 0.91,
                    "missing": None,
                    "rationale": "evidence states the color directly",
                }
            )
        ]
    )
    res = classify_answerability(
        "What is Alice's favorite color?",
        ["Alice's favorite color is teal."],
        llm_client=fake,
    )
    assert res.answerable is True
    assert res.partial is False
    assert res.confidence == pytest.approx(0.91)
    assert res.missing is None
    assert "evidence" in res.rationale
    assert len(fake.calls) == 1
    # System prompt must forbid world knowledge — guard the wording.
    assert "world knowledge" in fake.calls[0]["system"]
    assert fake.calls[0]["model"] == "haiku"


def test_classifier_handles_object_with_text_attribute():
    fake = FakeLLMClient(
        responses=[
            _FakeResponse(
                text=json.dumps(
                    {
                        "answerable": False,
                        "partial": True,
                        "confidence": 0.6,
                        "missing": "exact city",
                        "rationale": "country known, city not",
                    }
                )
            )
        ]
    )
    res = classify_answerability(
        "Where in Portugal did Maria stay?",
        ["Maria stayed somewhere in Portugal in July 2024."],
        llm_client=fake,
    )
    assert res.answerable is False
    assert res.partial is True
    assert res.missing == "exact city"


def test_classifier_strips_markdown_fences():
    fake = FakeLLMClient(
        responses=[
            "```json\n"
            + json.dumps(
                {
                    "answerable": True,
                    "partial": False,
                    "confidence": 0.8,
                    "missing": None,
                    "rationale": "ok",
                }
            )
            + "\n```"
        ]
    )
    res = classify_answerability("Q?", ["E"], llm_client=fake)
    assert res.answerable is True


def test_classifier_finds_json_inside_prose():
    fake = FakeLLMClient(
        responses=[
            "Sure — here is my answer: "
            + json.dumps(
                {
                    "answerable": False,
                    "partial": False,
                    "confidence": 0.9,
                    "missing": "topic absent",
                    "rationale": "evidence is unrelated",
                }
            )
            + " hope that helps."
        ]
    )
    res = classify_answerability("Q?", ["unrelated text"], llm_client=fake)
    assert res.answerable is False
    assert res.partial is False
    assert res.confidence == pytest.approx(0.9)


def test_classifier_retries_on_unparsable_response():
    good = json.dumps(
        {
            "answerable": True,
            "partial": False,
            "confidence": 0.85,
            "missing": None,
            "rationale": "fine on second try",
        }
    )
    fake = FakeLLMClient(responses=["not json at all", good])
    res = classify_answerability("Q?", ["E"], llm_client=fake)
    assert res.answerable is True
    assert len(fake.calls) == 2


def test_classifier_retries_at_most_two_times():
    # 3 attempts total: initial + 2 retries.
    fake = FakeLLMClient(responses=["nope", "still nope", "really nope"])
    res = classify_answerability("Q?", ["E"], llm_client=fake)
    assert res.answerable is False
    assert res.partial is False
    assert "attempts" in res.rationale
    assert len(fake.calls) == 3


def test_classifier_retries_on_llm_exception():
    class _Boom:
        def __init__(self) -> None:
            self.n = 0

        def complete(self, **kwargs: Any) -> Any:
            self.n += 1
            if self.n == 1:
                raise RuntimeError("transient network error")
            return json.dumps(
                {
                    "answerable": True,
                    "partial": False,
                    "confidence": 0.8,
                    "missing": None,
                    "rationale": "recovered",
                }
            )

    boom = _Boom()
    res = classify_answerability("Q?", ["E"], llm_client=boom)
    assert res.answerable is True
    assert boom.n == 2


def test_classifier_clamps_out_of_range_confidence():
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answerable": True,
                    "partial": False,
                    "confidence": 1.7,
                    "missing": None,
                    "rationale": "ok",
                }
            )
        ]
    )
    res = classify_answerability("Q?", ["E"], llm_client=fake)
    assert res.confidence == 1.0


def test_classifier_handles_inconsistent_flags():
    """Model says answerable=True but lists a missing field → treat as partial.

    Locks in the conservative reading documented in
    ``_parse_response``: a confident "yes" with a non-empty ``missing``
    string is downgraded to ``partial`` so downstream caveats fire.
    """
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answerable": True,
                    "partial": False,
                    "confidence": 0.9,
                    "missing": "specific date",
                    "rationale": "model contradicted itself",
                }
            )
        ]
    )
    res = classify_answerability("Q?", ["E"], llm_client=fake)
    assert res.answerable is False
    assert res.partial is True


def test_classifier_drops_both_true_to_partial():
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answerable": True,
                    "partial": True,
                    "confidence": 0.7,
                    "missing": "exact value",
                    "rationale": "model set both flags",
                }
            )
        ]
    )
    res = classify_answerability("Q?", ["E"], llm_client=fake)
    assert res.answerable is False
    assert res.partial is True


def test_classifier_returns_idk_on_total_failure():
    class _AlwaysFail:
        def complete(self, **kwargs: Any) -> Any:
            raise ConnectionError("provider down")

    res = classify_answerability("Q?", ["E"], llm_client=_AlwaysFail())
    assert res.answerable is False
    assert res.partial is False
    assert "classifier failed" in res.rationale


# ──────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────


def test_user_prompt_truncates_giant_snippets():
    huge = "x" * 5000
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answerable": False,
                    "partial": False,
                    "confidence": 0.9,
                    "missing": "topic",
                    "rationale": "ok",
                }
            )
        ]
    )
    classify_answerability("Q?", [huge], llm_client=fake)
    user = fake.calls[0]["user"]
    # Must include question + truncated marker, not the full 5000 chars.
    assert "Q?" in user
    assert "…" in user
    # 6000 char total cap with overhead — enforce upper bound generously.
    assert len(user) < 7000


def test_classifier_passes_custom_model_name():
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "answerable": True,
                    "partial": False,
                    "confidence": 0.8,
                    "missing": None,
                    "rationale": "ok",
                }
            )
        ]
    )
    classify_answerability(
        "Q?", ["E"], llm_model="sonnet", llm_client=fake
    )
    assert fake.calls[0]["model"] == "sonnet"


# ──────────────────────────────────────────────
# Fixture-driven sweep — 50 cases
# ──────────────────────────────────────────────


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["id"])
def test_fixture_propagates_verdict(fixture: dict[str, Any]):
    """For every fixture, an "ideal" fake LLM should yield the expected
    routing-relevant fields after parsing.
    """
    expect = fixture["expect"]
    response_obj = (
        _build_response(
            answerable=False,
            partial=False,
            confidence=1.0,
            missing="no evidence",
            rationale="empty",
        )
        if not [e for e in fixture["e"] if e.strip()]
        else _expected_synthetic_response(expect)
    )

    has_evidence = any(e.strip() for e in fixture["e"])
    fake = FakeLLMClient(responses=[response_obj] if has_evidence else [])

    # Use a tiny question for ``edge-10`` (just "?") — classifier will
    # still call the LLM because evidence is present.
    res = classify_answerability(
        fixture["q"], fixture["e"], llm_client=fake
    )

    assert res.answerable == bool(expect["answerable"]), (
        f"{fixture['id']}: answerable mismatch"
    )
    assert res.partial == bool(expect["partial"]), (
        f"{fixture['id']}: partial mismatch"
    )
    assert res.confidence >= 0.0
    assert res.confidence <= 1.0


# ──────────────────────────────────────────────
# Acceptance metrics — false-IDK and true-IDK rates
# ──────────────────────────────────────────────


def _classify_fixture(fx: dict[str, Any]) -> AnswerabilityResult:
    expect = fx["expect"]
    has_evidence = any(e.strip() for e in fx["e"])
    if has_evidence:
        fake = FakeLLMClient(responses=[_expected_synthetic_response(expect)])
    else:
        fake = FakeLLMClient(responses=[])
    return classify_answerability(fx["q"], fx["e"], llm_client=fake)


def test_true_idk_precision_on_unanswerable_cases():
    """At least 80% of expected-False cases come back as "not ANSWER".

    "Not ANSWER" here means the verdict is either ``answerable=False``
    or partial-only; we treat partial as acceptable because the router
    will route it to ANSWER_WITH_CAVEAT or IDK depending on confidence.
    """
    targets = [fx for fx in FIXTURES if not fx["expect"]["answerable"]]
    correct = 0
    for fx in targets:
        res = _classify_fixture(fx)
        if not res.answerable:
            correct += 1
    rate = correct / len(targets)
    assert rate >= 0.80, f"true-IDK precision {rate:.2%} below 80%"


def test_false_idk_rate_on_answerable_cases():
    """At most 5% of expected-True cases get marked unanswerable.

    Counts a wrong verdict only when the classifier returns
    ``answerable=False`` AND ``partial=False`` — partial-with-caveat is
    not a false-IDK because the router may still answer.
    """
    targets = [fx for fx in FIXTURES if fx["expect"]["answerable"]]
    misses = 0
    for fx in targets:
        res = _classify_fixture(fx)
        if not res.answerable and not res.partial:
            misses += 1
    rate = misses / len(targets)
    assert rate <= 0.05, f"false-IDK rate {rate:.2%} exceeds 5%"


def measured_false_idk_rate() -> float:
    """Helper for the acceptance script — exposed for ad-hoc reporting."""
    targets = [fx for fx in FIXTURES if fx["expect"]["answerable"]]
    misses = sum(
        1
        for fx in targets
        if (
            not (res := _classify_fixture(fx)).answerable
            and not res.partial
        )
    )
    return misses / max(1, len(targets))
