"""v9.0 D7 — judge-weighted ensemble picker."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

import ensemble_judge as ej  # noqa: E402


class _FakeClient:
    def __init__(self, response_text: str = ""):
        self._response_text = response_text
        self.calls: list[dict] = []
        self.should_raise: bool = False

    def complete(self, system: str, user: str, *, model: str, max_tokens: int):
        if self.should_raise:
            raise RuntimeError("simulated outage")
        self.calls.append(
            {"system": system, "user": user, "model": model, "max_tokens": max_tokens}
        )
        return SimpleNamespace(
            text=self._response_text, input_tokens=11, output_tokens=22
        )


def test_picks_highest_score():
    client = _FakeClient(json.dumps({"scores": [3.0, 8.5, 5.0], "abstain": False, "reason": "best b"}))
    pick = ej.judge_weighted_pick(
        client,
        question="Where did Alice go?",
        candidates=["Berlin", "Paris", "Rome"],
        category=1,
        judge_model="gpt-4o-mini",
    )
    assert pick.answer == "Paris"
    assert pick.scores == [3.0, 8.5, 5.0]
    assert pick.abstain is False
    assert pick.reason == "best b"
    assert pick.judge_input_tokens == 11
    assert pick.judge_output_tokens == 22


def test_abstain_flag_returns_default_refusal():
    client = _FakeClient(json.dumps({"scores": [2.0, 1.0], "abstain": True, "reason": "all weak"}))
    pick = ej.judge_weighted_pick(
        client,
        question="What did X say?",
        candidates=["foo", "bar"],
        category=5,
        judge_model="gpt-4o-mini",
    )
    assert pick.abstain is True
    assert pick.answer == "Not mentioned in the conversation."


def test_below_floor_triggers_abstain():
    client = _FakeClient(json.dumps({"scores": [2.0, 1.0], "abstain": False, "reason": "weak"}))
    pick = ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=["a", "b"],
        category=1,
        judge_model="gpt-4o-mini",
        min_score_floor=3.0,
    )
    assert pick.abstain is True
    assert pick.answer == "Not mentioned in the conversation."


def test_custom_abstain_answer():
    client = _FakeClient(json.dumps({"scores": [], "abstain": True}))
    pick = ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=["a", "b"],
        category=5,
        judge_model="gpt-4o-mini",
        abstain_answer="N/A",
    )
    assert pick.answer == "N/A"


def test_single_candidate_short_circuits_no_call():
    client = _FakeClient()
    pick = ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=["only one"],
        category=1,
        judge_model="m",
    )
    assert pick.answer == "only one"
    assert pick.scores == [10.0]
    assert pick.reason == "single_candidate"
    assert client.calls == []


def test_empty_candidates_returns_abstain():
    client = _FakeClient()
    pick = ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=[],
        category=5,
        judge_model="m",
    )
    assert pick.abstain is True
    assert pick.reason == "no_candidates"


def test_judge_call_failure_falls_back_to_first_candidate():
    client = _FakeClient()
    client.should_raise = True
    pick = ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=["first", "second"],
        category=1,
        judge_model="m",
    )
    assert pick.answer == "first"
    assert pick.reason == "judge_call_failed"


def test_parse_response_handles_garbage():
    scores, abstain, reason = ej._parse_response("nope not json", 3)
    assert scores == [5.0, 5.0, 5.0]
    assert abstain is False
    assert reason == "parse_error"


def test_parse_response_pads_short_score_list():
    raw = json.dumps({"scores": [4.0], "abstain": False, "reason": "ok"})
    scores, _, _ = ej._parse_response(raw, 3)
    assert scores == [4.0, 5.0, 5.0]


def test_parse_response_clamps_out_of_range():
    raw = json.dumps({"scores": [-2.0, 100.0, 5.0], "abstain": False})
    scores, _, _ = ej._parse_response(raw, 3)
    assert scores == [0.0, 10.0, 5.0]


def test_user_prompt_uses_category_rubric():
    client = _FakeClient(json.dumps({"scores": [9.0, 1.0], "abstain": False}))
    ej.judge_weighted_pick(
        client,
        question="When did X happen?",
        candidates=["May 2023", "today"],
        category=2,
        judge_model="m",
    )
    user = client.calls[0]["user"]
    assert "Temporal:" in user
    assert "QUESTION: When did X happen?" in user
    assert "[0] May 2023" in user


def test_unknown_category_uses_default_rubric():
    client = _FakeClient(json.dumps({"scores": [9.0, 1.0], "abstain": False}))
    ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=["a", "b"],
        category=99,
        judge_model="m",
    )
    user = client.calls[0]["user"]
    assert "Reward concise answers grounded" in user


def test_strips_fences_in_judge_response():
    client = _FakeClient(
        "```json\n" + json.dumps({"scores": [9.0, 1.0], "abstain": False}) + "\n```"
    )
    pick = ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=["A", "B"],
        category=1,
        judge_model="m",
    )
    assert pick.answer == "A"


def test_tie_break_prefers_earlier_candidate():
    client = _FakeClient(json.dumps({"scores": [7.0, 7.0, 7.0], "abstain": False}))
    pick = ej.judge_weighted_pick(
        client,
        question="Q?",
        candidates=["x", "y", "z"],
        category=1,
        judge_model="m",
    )
    assert pick.answer == "x"
