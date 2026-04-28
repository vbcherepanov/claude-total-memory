"""Unit tests for ``memory_core.negative_retrieval`` (W2-H, v11).

Negative retrieval performs a *second* pass against an inverted query
to surface contradicting evidence. The whole point is to make IDK
decisions robust on adversarial LoCoMo questions where positive
retrieval alone hallucinates.

Every test mocks the LLM (``FakeLLMClient``), the search backend
(``FakeSearch``), and the contradiction scorer (a deterministic lambda)
— no test reaches the network or hits a real DB.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# Make src/ importable on its own.
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory_core.negative_retrieval import (  # noqa: E402
    NegativeEvidenceResult,
    THRESHOLD_HARD,
    THRESHOLD_SOFT,
    negative_retrieve,
)


FIXTURES_PATH = (
    Path(__file__).parent / "fixtures" / "negative_retrieval_fixtures.json"
)


# ──────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────


@dataclass
class _FakeResponse:
    """Mimic ``LLMResult`` so the module's ``.text`` extraction is exercised."""

    text: str


class FakeLLMClient:
    """FIFO queue of canned responses. Empty queue raises if hit."""

    def __init__(self, responses: list[Any]) -> None:
        self.q: list[Any] = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> Any:
        if not self.q:
            raise AssertionError("FakeLLMClient response queue exhausted")
        self.calls.append(kwargs)
        return self.q.pop(0)


class FakeSearch:
    """Records calls and returns a queue of pre-baked hit lists."""

    def __init__(self, returns: list[list[dict]]) -> None:
        self._returns: list[list[dict]] = list(returns)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        query: str,
        k: int = 10,
        project: str | None = None,
    ) -> list[dict]:
        self.calls.append({"query": query, "k": k, "project": project})
        if not self._returns:
            return []
        return self._returns.pop(0)


def _contradict_by_keyword(fact_a: str, fact_b: str) -> float:
    """Deterministic contradiction scorer used across the suite.

    The contract from the task brief: ``0.7 if "not" in b else 0.1``.
    We extend it with a "soft" middle band so we can exercise all three
    decision branches without a real model: words like ``"only"`` /
    ``"few"`` / ``"prototype"`` move the score into the soft zone.
    """
    b = (fact_b or "").lower()
    if "not " in b or " no " in b or "cancelled" in b or "cancel" in b:
        return 0.7
    soft_markers = (
        "only ",
        "few ",
        "overrated",
        "prototype",
        "still running",
        "half",
    )
    if any(marker in b for marker in soft_markers):
        return 0.45
    return 0.1


def _zero_contradict(fact_a: str, fact_b: str) -> float:
    return 0.0


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _load_fixtures() -> dict[str, list[dict]]:
    with FIXTURES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


FIXTURES = _load_fixtures()


def _build_inverted(query_line: str) -> str:
    """Produce a single-line response the module will accept."""
    return query_line


# ──────────────────────────────────────────────
# Fixture sanity
# ──────────────────────────────────────────────


def test_fixtures_have_expected_buckets():
    assert set(FIXTURES.keys()) == {
        "hard_contradict",
        "no_contradiction",
        "soft_contradict",
        "adversarial",
    }
    assert len(FIXTURES["hard_contradict"]) == 5
    assert len(FIXTURES["no_contradiction"]) == 5
    assert len(FIXTURES["soft_contradict"]) == 5
    assert len(FIXTURES["adversarial"]) >= 3


# ──────────────────────────────────────────────
# Edge cases — no LLM call expected
# ──────────────────────────────────────────────


def test_empty_positive_evidence_short_circuits_without_llm_or_search():
    fake_llm = FakeLLMClient(responses=[])
    fake_search = FakeSearch(returns=[])
    res = negative_retrieve(
        "What is Alice's color?",
        [],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert isinstance(res, NegativeEvidenceResult)
    assert res.decision == "no_contradiction"
    assert res.contradiction_score == 0.0
    assert res.negative_evidence == []
    assert res.inverted_query == ""
    assert "no positive evidence" in res.rationale
    assert fake_llm.calls == []
    assert fake_search.calls == []


def test_positives_with_only_blank_text_treated_as_empty():
    fake_llm = FakeLLMClient(responses=[])
    fake_search = FakeSearch(returns=[])
    res = negative_retrieve(
        "Q?",
        [{"text": ""}, {"text": "   "}, {"content": None}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.decision == "no_contradiction"
    assert fake_llm.calls == []
    assert fake_search.calls == []


def test_empty_question_short_circuits_cleanly():
    fake_llm = FakeLLMClient(responses=[])
    fake_search = FakeSearch(returns=[])
    res = negative_retrieve(
        "   ",
        [{"text": "some positive fact"}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.decision == "no_contradiction"
    assert "empty question" in res.rationale
    assert fake_llm.calls == []
    assert fake_search.calls == []


# ──────────────────────────────────────────────
# Inverted-query handling
# ──────────────────────────────────────────────


def test_inverted_query_takes_first_non_empty_line():
    multiline = "\n\n   \nfacts denying Alice loves teal\nsecond line\n"
    fake_llm = FakeLLMClient(responses=[multiline])
    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "What is Alice's favorite color?",
        [{"text": "Alice's favorite color is teal."}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.inverted_query == "facts denying Alice loves teal"
    assert len(fake_llm.calls) == 1
    assert fake_search.calls[0]["query"] == "facts denying Alice loves teal"


def test_inverted_query_strips_markdown_fence_and_quotes():
    raw = '```\n"facts contradicting Alice loves teal"\n```'
    fake_llm = FakeLLMClient(responses=[raw])
    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "What is Alice's favorite color?",
        [{"text": "Alice loves teal."}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.inverted_query == "facts contradicting Alice loves teal"


def test_inverted_query_retries_on_empty_first_attempt():
    fake_llm = FakeLLMClient(responses=["", "actual contradicting query"])
    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "Q? what about X",
        [{"text": "X is true"}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.inverted_query == "actual contradicting query"
    assert len(fake_llm.calls) == 2


def test_inverted_query_retries_when_model_echoes_question():
    fake_llm = FakeLLMClient(
        responses=[
            "What is Alice's favorite color?",
            "facts denying Alice's color preference",
        ]
    )
    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "What is Alice's favorite color?",
        [{"text": "Alice loves teal."}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.inverted_query == "facts denying Alice's color preference"
    assert len(fake_llm.calls) == 2


def test_inverted_query_falls_back_to_template_on_llm_failures():
    class _AlwaysRaise:
        def complete(self, **kwargs: Any) -> Any:
            raise RuntimeError("provider down")

    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "What is Alice's favorite color?",
        [{"text": "Alice loves teal."}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=_AlwaysRaise(),
    )
    assert "contradicting" in res.inverted_query.lower()
    assert "Alice" in res.inverted_query
    assert fake_search.calls[0]["query"] == res.inverted_query


def test_inverted_query_uses_template_when_no_llm_client():
    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "Where is Maria?",
        [{"text": "Maria is in Lisbon."}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=None,
    )
    assert "Maria" in res.inverted_query
    assert "contradicting" in res.inverted_query.lower()


def test_llm_response_with_text_attribute_handled():
    fake_llm = FakeLLMClient(
        responses=[_FakeResponse(text="contradicting Alice loves teal")]
    )
    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "What is Alice's favorite color?",
        [{"text": "Alice loves teal."}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.inverted_query == "contradicting Alice loves teal"


def test_llm_call_uses_haiku_model_by_default():
    fake_llm = FakeLLMClient(responses=["query"])
    fake_search = FakeSearch(returns=[[]])
    negative_retrieve(
        "Q?",
        [{"text": "fact"}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert fake_llm.calls[0]["model"] == "haiku"
    assert "CONTRADICT" in fake_llm.calls[0]["system"]


def test_llm_call_passes_custom_model_alias():
    fake_llm = FakeLLMClient(responses=["query"])
    fake_search = FakeSearch(returns=[[]])
    negative_retrieve(
        "Q?",
        [{"text": "fact"}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
        llm_model="sonnet",
    )
    assert fake_llm.calls[0]["model"] == "sonnet"


# ──────────────────────────────────────────────
# Search behaviour
# ──────────────────────────────────────────────


def test_search_fn_receives_project_and_k():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[]])
    negative_retrieve(
        "Q?",
        [{"text": "fact"}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        project="vito",
        k=7,
        llm_client=fake_llm,
    )
    assert fake_search.calls[0]["project"] == "vito"
    assert fake_search.calls[0]["k"] == 7


def test_search_failure_is_swallowed_into_no_contradiction():
    class _Boom:
        def __call__(self, query: str, k: int = 10, project: str | None = None) -> list[dict]:
            raise ConnectionError("vector store offline")

    fake_llm = FakeLLMClient(responses=["q"])
    res = negative_retrieve(
        "Q?",
        [{"text": "fact"}],
        search_fn=_Boom(),
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.decision == "no_contradiction"
    assert "negative search failed" in res.rationale


def test_negative_search_no_hits_returns_no_contradiction():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[]])
    res = negative_retrieve(
        "Q?",
        [{"text": "fact"}],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )
    assert res.decision == "no_contradiction"
    assert "no hits" in res.rationale
    assert res.contradiction_score == 0.0


# ──────────────────────────────────────────────
# Decision-threshold logic
# ──────────────────────────────────────────────


def test_decision_thresholds_match_spec():
    assert THRESHOLD_SOFT == pytest.approx(0.30)
    assert THRESHOLD_HARD == pytest.approx(0.60)


def test_constant_zero_score_yields_no_contradiction():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[{"text": "anything"}]])
    res = negative_retrieve(
        "Q?",
        [{"text": "fact"}],
        search_fn=fake_search,
        contradiction_fn=_zero_contradict,
        llm_client=fake_llm,
    )
    assert res.decision == "no_contradiction"
    assert res.contradiction_score == pytest.approx(0.0)


def test_score_just_above_soft_threshold_is_soft_contradict():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[{"text": "neg"}]])
    res = negative_retrieve(
        "Q?",
        [{"text": "pos"}],
        search_fn=fake_search,
        contradiction_fn=lambda a, b: 0.31,
        llm_client=fake_llm,
    )
    assert res.decision == "soft_contradict"
    assert res.contradiction_score == pytest.approx(0.31)


def test_score_just_below_hard_threshold_is_soft_contradict():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[{"text": "neg"}]])
    res = negative_retrieve(
        "Q?",
        [{"text": "pos"}],
        search_fn=fake_search,
        contradiction_fn=lambda a, b: 0.59,
        llm_client=fake_llm,
    )
    assert res.decision == "soft_contradict"


def test_score_at_hard_threshold_is_hard_contradict():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[{"text": "neg"}]])
    res = negative_retrieve(
        "Q?",
        [{"text": "pos"}],
        search_fn=fake_search,
        contradiction_fn=lambda a, b: 0.6,
        llm_client=fake_llm,
    )
    assert res.decision == "hard_contradict"


def test_max_score_taken_across_pairs():
    """Best-of-pairs aggregation — even one strong conflict trips the verdict."""
    scores = iter([0.05, 0.05, 0.72, 0.05])
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(
        returns=[[{"text": "n1"}, {"text": "n2"}]]
    )
    res = negative_retrieve(
        "Q?",
        [{"text": "p1"}, {"text": "p2"}],
        search_fn=fake_search,
        contradiction_fn=lambda a, b: next(scores),
        llm_client=fake_llm,
    )
    assert res.decision == "hard_contradict"
    assert res.contradiction_score == pytest.approx(0.72)


def test_pairwise_evaluation_caps_at_5x5():
    """Both sides clipped to 5 entries → at most 25 contradiction calls."""
    call_count = {"n": 0}

    def counting_fn(a: str, b: str) -> float:
        call_count["n"] += 1
        return 0.05

    pos = [{"text": f"p{i}"} for i in range(10)]
    neg = [{"text": f"n{i}"} for i in range(10)]
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[neg])
    negative_retrieve(
        "Q?",
        pos,
        search_fn=fake_search,
        contradiction_fn=counting_fn,
        llm_client=fake_llm,
    )
    assert call_count["n"] == 25


def test_contradiction_fn_exception_swallowed_per_pair():
    """A single bad scoring call must not crash the whole pass."""
    seq = iter([0.0, 0.0])

    def flaky(a: str, b: str) -> float:
        # First pair raises, second returns 0.45 → soft band.
        if "boom" in b:
            raise RuntimeError("scorer hiccup")
        return next(seq) if False else 0.45

    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(
        returns=[[{"text": "boom-neg"}, {"text": "calm-neg"}]]
    )
    res = negative_retrieve(
        "Q?",
        [{"text": "pos"}],
        search_fn=fake_search,
        contradiction_fn=flaky,
        llm_client=fake_llm,
    )
    assert res.decision == "soft_contradict"
    assert res.contradiction_score == pytest.approx(0.45)


def test_contradiction_score_clamped_to_unit_interval():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[{"text": "neg"}]])
    res = negative_retrieve(
        "Q?",
        [{"text": "pos"}],
        search_fn=fake_search,
        contradiction_fn=lambda a, b: 5.0,
        llm_client=fake_llm,
    )
    assert res.contradiction_score == pytest.approx(1.0)
    assert res.decision == "hard_contradict"


def test_rationale_contains_pair_preview_for_contradicting_decision():
    fake_llm = FakeLLMClient(responses=["q"])
    fake_search = FakeSearch(returns=[[{"text": "Alice does NOT like teal."}]])
    res = negative_retrieve(
        "What is Alice's favorite color?",
        [{"text": "Alice loves teal."}],
        search_fn=fake_search,
        contradiction_fn=lambda a, b: 0.8,
        llm_client=fake_llm,
    )
    assert "Alice" in res.rationale
    assert "teal" in res.rationale.lower()
    assert "0.80" in res.rationale or "0.8" in res.rationale


# ──────────────────────────────────────────────
# Fixture-driven sweep — 5 hard / 5 soft / 5 none / 3 adversarial
# ──────────────────────────────────────────────


def _run_with_canned(fixture: dict, expected_decision: str) -> NegativeEvidenceResult:
    """Run negative_retrieve with the fixture's canned negatives.

    The LLM response is the fixture's pre-baked inverted query so the
    test stays deterministic. The search backend returns the fixture's
    canned negatives. The contradiction scorer is the keyword stub.
    """
    inverted = fixture.get("inverted_query") or "contradicting facts"
    fake_llm = FakeLLMClient(responses=[_build_inverted(inverted)])
    fake_search = FakeSearch(returns=[fixture["negative_canned"]])
    return negative_retrieve(
        fixture["q"],
        fixture["positive"],
        search_fn=fake_search,
        contradiction_fn=_contradict_by_keyword,
        llm_client=fake_llm,
    )


@pytest.mark.parametrize(
    "fixture",
    FIXTURES["hard_contradict"],
    ids=lambda f: f["id"],
)
def test_fixture_hard_contradict(fixture: dict):
    res = _run_with_canned(fixture, "hard_contradict")
    assert res.decision == "hard_contradict", (
        f"{fixture['id']}: got {res.decision} score={res.contradiction_score:.2f}"
    )
    assert res.contradiction_score >= THRESHOLD_HARD


@pytest.mark.parametrize(
    "fixture",
    FIXTURES["no_contradiction"],
    ids=lambda f: f["id"],
)
def test_fixture_no_contradiction(fixture: dict):
    res = _run_with_canned(fixture, "no_contradiction")
    assert res.decision == "no_contradiction", (
        f"{fixture['id']}: got {res.decision} score={res.contradiction_score:.2f}"
    )
    assert res.contradiction_score < THRESHOLD_SOFT


@pytest.mark.parametrize(
    "fixture",
    FIXTURES["soft_contradict"],
    ids=lambda f: f["id"],
)
def test_fixture_soft_contradict(fixture: dict):
    res = _run_with_canned(fixture, "soft_contradict")
    assert res.decision == "soft_contradict", (
        f"{fixture['id']}: got {res.decision} score={res.contradiction_score:.2f}"
    )
    assert THRESHOLD_SOFT <= res.contradiction_score < THRESHOLD_HARD


@pytest.mark.parametrize(
    "fixture",
    FIXTURES["adversarial"],
    ids=lambda f: f["id"],
)
def test_fixture_adversarial_blocks_hallucination(fixture: dict):
    """Adversarial: positives look supportive, negatives reveal absence.

    All three adversarial fixtures expect ``hard_contradict`` because
    the negative side carries an explicit "not" / "does not" / "did
    not" denial. This proves the negative pass would force IDK in
    cases where the positive-only path would have hallucinated a
    confident answer.
    """
    expected = fixture.get("expected_decision", "hard_contradict")
    res = _run_with_canned(fixture, expected)
    assert res.decision == expected, (
        f"{fixture['id']}: got {res.decision} score={res.contradiction_score:.2f}"
    )


# ──────────────────────────────────────────────
# Layer-wall regression
# ──────────────────────────────────────────────


def test_module_does_not_import_ai_layer():
    """The whole point of putting this in memory_core: layer wall."""
    import ast

    path = SRC / "memory_core" / "negative_retrieval.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "ai_layer":
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                if node.module.split(".")[0] == "ai_layer":
                    offending.append(node.module)
    assert offending == [], f"ai_layer imports forbidden in memory_core: {offending}"
