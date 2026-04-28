"""Tests for ai_layer.iterative_retriever (IRCoT loop).

All LLM and search dependencies are mocked — no real network. The
planner LLM responses are pre-queued; the search function is a
deterministic in-memory fake.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make src/ importable for the in-source layout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_layer import iterative_retriever as ir  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────


class FakeLLMResult:
    """Minimal LLMResult-like container."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.input_tokens = 0
        self.output_tokens = 0


class FakeLLMClient:
    """Returns queued strings as LLMResult.text. Tracks every call."""

    def __init__(self, responses: list[str]) -> None:
        self._queue = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retries: int = 3,
    ) -> FakeLLMResult:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self._queue:
            raise RuntimeError("FakeLLMClient: response queue exhausted")
        return FakeLLMResult(self._queue.pop(0))


class FakeSearch:
    """Returns deterministic dicts based on a per-query script.

    ``script`` maps a substring matcher -> list of hit dicts. Falls back
    to a single placeholder hit if no matcher hits, so the planner still
    sees something.
    """

    def __init__(self, script: dict[str, list[dict]] | None = None) -> None:
        self.script = script or {}
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        query: str,
        k: int = 10,
        project: str | None = None,
    ) -> list[dict]:
        self.calls.append({"query": query, "k": k, "project": project})
        for needle, hits in self.script.items():
            if needle.lower() in query.lower():
                return [dict(h) for h in hits[:k]]
        # Default: synthesize a single hit derived from the query so we
        # still feed the planner deterministically.
        return [
            {"id": f"default::{query[:32]}", "content": f"placeholder for {query}"}
        ]


def _planner_json(partial: str, next_q: str | None, done: bool) -> str:
    return json.dumps({"partial_answer": partial, "next_query": next_q, "done": done})


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_rewrite(monkeypatch):
    """Replace ai_layer.iterative_retriever._rewrite with a stub.

    Yields a setter that the test calls with the desired rewrite output.
    """
    state: dict[str, dict] = {"value": {"canonical": "", "decomposed": [], "hyde": ""}}

    def _setter(*, canonical: str = "", decomposed: list[str] | None = None, hyde: str = "") -> None:
        state["value"] = {
            "canonical": canonical,
            "decomposed": list(decomposed or []),
            "hyde": hyde,
        }

    def _stub(query: str, **_kwargs):
        v = state["value"]
        return {
            "canonical": v["canonical"] or query,
            "decomposed": list(v["decomposed"]),
            "hyde": v["hyde"],
        }

    monkeypatch.setattr(ir, "_rewrite", _stub)
    return _setter


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def test_decomposes_multihop(fake_rewrite):
    """Multi-hop question dispatches one search per decomposed sub-query.

    With 2 decomposed sub-queries and a planner that says ``done`` only
    on the second iteration, we expect at least 2 search calls.
    """
    fake_rewrite(
        canonical="Alice Bob Charlie project",
        decomposed=[
            "What did Alice tell Bob",
            "Charlie's project details",
        ],
    )
    search = FakeSearch(
        {
            "alice": [{"id": "k1", "content": "Alice told Bob about the deadline."}],
            "charlie": [{"id": "k2", "content": "Charlie ships the parser module."}],
        }
    )
    llm = FakeLLMClient(
        [
            _planner_json("Alice mentioned a deadline.", "Charlie's project details", False),
            _planner_json("Charlie ships the parser.", None, True),
        ]
    )

    result = ir.iterative_retrieve(
        "What did Alice tell Bob about Charlie's project?",
        search_fn=search,
        max_iters=4,
        k_per_iter=3,
        llm_client=llm,
    )

    assert len(search.calls) >= 2
    assert result.iterations_used >= 2
    assert result.terminated_reason == "converged"
    issued = [c["query"] for c in search.calls]
    assert "What did Alice tell Bob" in issued
    assert "Charlie's project details" in issued


def test_terminates_on_done(fake_rewrite):
    """Planner returns done=true on iter 2 -> exactly 2 retrieval calls."""
    fake_rewrite(canonical="q1", decomposed=["q1", "q2", "q3"])
    search = FakeSearch()
    llm = FakeLLMClient(
        [
            _planner_json("partial 1", "q2", False),
            _planner_json("partial 2 final", None, True),
            # Anything beyond should not be consumed.
            _planner_json("never", None, True),
        ]
    )

    result = ir.iterative_retrieve(
        "user question",
        search_fn=search,
        max_iters=5,
        k_per_iter=2,
        llm_client=llm,
    )

    assert len(search.calls) == 2
    assert result.iterations_used == 2
    assert result.terminated_reason == "converged"
    assert result.partial_answers == ["partial 1", "partial 2 final"]


def test_max_iters_cap(fake_rewrite):
    """Planner never says done -> exactly max_iters retrievals."""
    fake_rewrite(canonical="seed", decomposed=["seed"])
    search = FakeSearch()
    # Always emit done=false with a fresh next_query.
    responses = [
        _planner_json(f"partial {i}", f"next-q-{i}", False) for i in range(10)
    ]
    llm = FakeLLMClient(responses)

    result = ir.iterative_retrieve(
        "user question",
        search_fn=search,
        max_iters=3,
        k_per_iter=2,
        llm_client=llm,
    )

    assert len(search.calls) == 3
    assert result.iterations_used == 3
    assert result.terminated_reason == "max_iters"


def test_dedup_evidence(fake_rewrite):
    """Same hit returned across iters -> appears once in final_evidence."""
    fake_rewrite(canonical="seed", decomposed=["q1", "q2"])
    duplicate_hit = {"id": "k-shared", "content": "Shared evidence A"}
    search = FakeSearch(
        {
            "q1": [duplicate_hit, {"id": "k-only-1", "content": "Unique 1"}],
            "q2": [duplicate_hit, {"id": "k-only-2", "content": "Unique 2"}],
        }
    )
    llm = FakeLLMClient(
        [
            _planner_json("p1", "q2", False),
            _planner_json("p2", None, True),
        ]
    )

    result = ir.iterative_retrieve(
        "anything",
        search_fn=search,
        max_iters=4,
        k_per_iter=5,
        llm_client=llm,
    )

    ids = [h["id"] for h in result.final_evidence]
    assert ids.count("k-shared") == 1
    assert set(ids) == {"k-shared", "k-only-1", "k-only-2"}


def test_robust_to_bad_json(fake_rewrite):
    """Malformed first JSON -> retry once, succeed on the second response."""
    fake_rewrite(canonical="seed", decomposed=["seed"])
    search = FakeSearch()
    llm = FakeLLMClient(
        [
            "not json at all {oops",
            _planner_json("recovered", None, True),
        ]
    )

    result = ir.iterative_retrieve(
        "anything",
        search_fn=search,
        max_iters=3,
        k_per_iter=2,
        llm_client=llm,
    )

    # Two LLM calls total within a single iteration (1 fail + 1 success).
    assert len(llm.calls) == 2
    # Only one retrieval performed before convergence.
    assert len(search.calls) == 1
    assert result.terminated_reason == "converged"
    iters = result.provenance["iters"]
    assert len(iters) == 1
    assert iters[0]["planner_parse_attempts"] == 2
    assert result.partial_answers == ["recovered"]


def test_decomposer_empty_falls_back_to_canonical(fake_rewrite):
    """Empty `decomposed` -> uses canonical as the single seed."""
    fake_rewrite(canonical="canonical-form", decomposed=[])
    search = FakeSearch(
        {"canonical-form": [{"id": "k-canon", "content": "Canonical evidence."}]}
    )
    llm = FakeLLMClient(
        [
            _planner_json("done immediately", None, True),
        ]
    )

    result = ir.iterative_retrieve(
        "raw user question",
        search_fn=search,
        max_iters=4,
        k_per_iter=3,
        llm_client=llm,
    )

    assert len(search.calls) == 1
    assert search.calls[0]["query"] == "canonical-form"
    assert result.iterations_used == 1
    assert result.terminated_reason == "converged"
    assert result.sub_queries == ["canonical-form"]
    assert result.provenance["rewrite"]["used_decomposition"] is False


def test_provenance_contains_per_iter_timing(fake_rewrite):
    """Provenance records per-iteration metadata."""
    fake_rewrite(canonical="seed", decomposed=["seed"])
    search = FakeSearch()
    llm = FakeLLMClient([_planner_json("p", None, True)])

    result = ir.iterative_retrieve(
        "q",
        search_fn=search,
        max_iters=2,
        k_per_iter=1,
        llm_client=llm,
    )

    assert "iters" in result.provenance
    assert len(result.provenance["iters"]) == 1
    iter0 = result.provenance["iters"][0]
    assert iter0["sub_query"] == "seed"
    assert iter0["hits_returned"] >= 0
    assert iter0["planner_done"] is True
    assert iter0["elapsed_ms"] >= 0
    assert "elapsed_ms" in result.provenance


def test_invalid_arguments(fake_rewrite):
    """Empty query and bad caps surface as ValueError."""
    fake_rewrite(canonical="", decomposed=[])
    search = FakeSearch()
    llm = FakeLLMClient([])

    with pytest.raises(ValueError):
        ir.iterative_retrieve("", search_fn=search, llm_client=llm)
    with pytest.raises(ValueError):
        ir.iterative_retrieve("q", search_fn=search, max_iters=0, llm_client=llm)
    with pytest.raises(ValueError):
        ir.iterative_retrieve("q", search_fn=search, k_per_iter=0, llm_client=llm)


def test_planner_violates_contract_treated_as_done(fake_rewrite):
    """next_query=null with done=false is coerced to done=true (safe stop)."""
    fake_rewrite(canonical="seed", decomposed=["seed"])
    search = FakeSearch()
    llm = FakeLLMClient(
        [
            _planner_json("contradictory", None, False),
            # Should NOT be consumed.
            _planner_json("never", "x", False),
        ]
    )

    result = ir.iterative_retrieve(
        "q",
        search_fn=search,
        max_iters=4,
        k_per_iter=1,
        llm_client=llm,
    )

    assert result.terminated_reason == "converged"
    assert result.iterations_used == 1
    assert len(llm.calls) == 1
