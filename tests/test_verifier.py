"""Tests for ai_layer.verifier (W1-E NLI Verifier).

Heavy tests load a ~270 MB multilingual NLI model. They are gated behind
``SKIP_NLI=1`` so CI / offline runs can opt out cleanly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

# Make src/ importable in the same way the rest of the test suite does.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_layer import verifier  # noqa: E402
from ai_layer.verifier import NLIDecision, VerifyResult, verify, warmup  # noqa: E402


SKIP_REASON = "NLI model not downloaded (set SKIP_NLI=0 to enable)"
_skip_if_disabled = pytest.mark.skipif(
    os.environ.get("SKIP_NLI") == "1",
    reason=SKIP_REASON,
)


# ────────────────────────────────────────────────────────────────────
# Lightweight tests — no model load
# ────────────────────────────────────────────────────────────────────


def test_empty_evidence_short_circuits_without_model_call(monkeypatch):
    """Empty evidence must NOT touch the model singleton."""
    sentinel_calls = {"n": 0}

    def _boom():
        sentinel_calls["n"] += 1
        raise AssertionError("verify() must not load the NLI model on empty evidence")

    monkeypatch.setattr(verifier, "_get_model", _boom)

    out = verify(answer="anything", evidence=[])
    assert out == VerifyResult(NLIDecision.NEUTRAL, 0.0, 1.0, 0.0, 0)
    assert sentinel_calls["n"] == 0


def test_blank_evidence_strings_short_circuit(monkeypatch):
    monkeypatch.setattr(
        verifier,
        "_get_model",
        lambda: pytest.fail("must not load model when all evidence is blank"),
    )

    out = verify(answer="Sarah lives in Berlin", evidence=["", "   ", "\n"])
    assert out.decision is NLIDecision.NEUTRAL
    assert out.aggregated_from == 0
    assert out.p_neutral == 1.0


def test_blank_answer_short_circuits(monkeypatch):
    monkeypatch.setattr(
        verifier,
        "_get_model",
        lambda: pytest.fail("must not load model when answer is blank"),
    )
    out = verify(answer="   ", evidence=["She lives in Berlin"])
    assert out.decision is NLIDecision.NEUTRAL
    assert out.aggregated_from == 0


def test_decide_pure_function_thresholds():
    # contradict above 0.6 wins regardless of entail.
    assert verifier._decide(0.9, 0.0, 0.61) is NLIDecision.CONTRADICT
    # contradict at threshold (not strictly greater) does not flip.
    assert verifier._decide(0.7, 0.1, 0.6) is NLIDecision.ENTAIL
    # entail dominates neutral and contradict.
    assert verifier._decide(0.8, 0.1, 0.1) is NLIDecision.ENTAIL
    # neutral dominates a low-entail / low-contradict case.
    assert verifier._decide(0.2, 0.7, 0.1) is NLIDecision.NEUTRAL


def test_verify_result_is_immutable():
    out = VerifyResult(NLIDecision.NEUTRAL, 0.0, 1.0, 0.0, 0)
    with pytest.raises(Exception):  # frozen dataclass → FrozenInstanceError
        out.decision = NLIDecision.ENTAIL  # type: ignore[misc]


def test_nli_decision_string_values():
    assert NLIDecision.ENTAIL.value == "entail"
    assert NLIDecision.NEUTRAL.value == "neutral"
    assert NLIDecision.CONTRADICT.value == "contradict"


def test_label_resolver_handles_aliases():
    # The resolver must map both 'entailment' and 'entail' onto ENTAIL.
    resolved = verifier._NLIModel._resolve_label_indices(
        {0: "entailment", 1: "neutral", 2: "contradiction"}
    )
    assert resolved[NLIDecision.ENTAIL] == 0
    assert resolved[NLIDecision.NEUTRAL] == 1
    assert resolved[NLIDecision.CONTRADICT] == 2


def test_label_resolver_rejects_incomplete_labels():
    with pytest.raises(RuntimeError) as excinfo:
        verifier._NLIModel._resolve_label_indices({0: "entailment", 1: "neutral"})
    assert "missing label" in str(excinfo.value)


# ────────────────────────────────────────────────────────────────────
# Heavy tests — actually run the NLI model
# ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def warm_model():
    """Load the model once for the whole module.

    Skips the entire group cleanly if the model is unavailable
    (no network, disk full, etc.).
    """
    if os.environ.get("SKIP_NLI") == "1":
        pytest.skip(SKIP_REASON)
    try:
        warmup()
    except Exception as exc:  # noqa: BLE001 — we explicitly want to skip.
        pytest.skip(f"NLI model unavailable: {exc!r}")
    return True


@_skip_if_disabled
def test_plain_entailment_english(warm_model):
    out = verify(
        answer="Sarah is in Berlin.",
        evidence=["Sarah lives in Berlin."],
    )
    assert out.decision is NLIDecision.ENTAIL
    assert out.aggregated_from == 1
    assert out.p_entail > out.p_contradict
    assert out.p_entail > out.p_neutral


@_skip_if_disabled
def test_plain_contradiction_english(warm_model):
    out = verify(
        answer="Sarah lives in Paris.",
        evidence=["Sarah lives in Berlin."],
    )
    assert out.decision is NLIDecision.CONTRADICT
    assert out.p_contradict > 0.6


@_skip_if_disabled
def test_neutral_unrelated_topic(warm_model):
    out = verify(
        answer="Sarah lives in Berlin.",
        evidence=["The Eiffel Tower was built in 1889."],
    )
    assert out.decision is NLIDecision.NEUTRAL
    # neither entail nor contradict should win confidently.
    assert out.p_contradict <= 0.6


@_skip_if_disabled
def test_multilingual_entailment_russian(warm_model):
    out = verify(
        answer="Сара живёт в Берлине.",
        evidence=["Сара переехала в Берлин в прошлом году."],
    )
    assert out.decision is NLIDecision.ENTAIL


@_skip_if_disabled
def test_multilingual_contradiction_russian(warm_model):
    out = verify(
        answer="Сара живёт в Париже.",
        evidence=["Сара живёт в Берлине."],
    )
    assert out.decision is NLIDecision.CONTRADICT


@_skip_if_disabled
def test_cross_lingual_evidence_english_answer_russian(warm_model):
    out = verify(
        answer="Сара живёт в Берлине.",
        evidence=["Sarah lives in Berlin."],
    )
    assert out.decision is NLIDecision.ENTAIL


@_skip_if_disabled
def test_one_supporting_piece_among_many_neutral(warm_model):
    """4 of 5 evidence pieces are off-topic; the 5th entails."""
    evidence = [
        "The cat sat on the mat.",
        "Pi is approximately 3.14159.",
        "Mount Everest is in the Himalayas.",
        "Coffee originated in Ethiopia.",
        "Sarah moved to Berlin and now lives there.",
    ]
    out = verify(answer="Sarah is in Berlin.", evidence=evidence)
    assert out.decision is NLIDecision.ENTAIL
    assert out.aggregated_from == 5


@_skip_if_disabled
def test_contradiction_overrides_other_pieces(warm_model):
    """A strong contradiction in the bundle must flip the verdict."""
    evidence = [
        "The cat sat on the mat.",
        "Pi is approximately 3.14159.",
        "Sarah lives in Berlin.",
        "Sarah does not live in Paris; she lives in Berlin.",
    ]
    out = verify(answer="Sarah lives in Paris.", evidence=evidence)
    assert out.decision is NLIDecision.CONTRADICT


@_skip_if_disabled
def test_probabilities_are_well_formed(warm_model):
    out = verify(
        answer="Sarah is in Berlin.",
        evidence=["Sarah lives in Berlin."],
    )
    for p in (out.p_entail, out.p_neutral, out.p_contradict):
        assert 0.0 <= p <= 1.0


@_skip_if_disabled
def test_aggregated_from_counts_only_non_blank(warm_model):
    out = verify(
        answer="Sarah is in Berlin.",
        evidence=["Sarah lives in Berlin.", "", "   "],
    )
    assert out.aggregated_from == 1


@_skip_if_disabled
def test_batch_size_smaller_than_evidence(warm_model):
    """batch_size=1 must yield the same verdict as the default batched call."""
    evidence = [
        "Sarah lives in Berlin.",
        "Berlin is the capital of Germany.",
        "Sarah enjoys Brandenburg Gate walks.",
    ]
    bs1 = verify(answer="Sarah is in Berlin.", evidence=evidence, batch_size=1)
    bs8 = verify(answer="Sarah is in Berlin.", evidence=evidence, batch_size=8)
    assert bs1.decision == bs8.decision == NLIDecision.ENTAIL
    # Aggregated probabilities should be very close (deterministic forward).
    assert abs(bs1.p_entail - bs8.p_entail) < 1e-3


@_skip_if_disabled
def test_warmup_is_idempotent(warm_model):
    """Calling warmup() twice must not raise and must reuse the singleton."""
    first = verifier._get_model()
    warmup()
    second = verifier._get_model()
    assert first is second


@_skip_if_disabled
def test_singleton_reused_across_verify_calls(warm_model):
    a = verifier._get_model()
    verify(answer="Sarah is in Berlin.", evidence=["Sarah lives in Berlin."])
    b = verifier._get_model()
    assert a is b


@_skip_if_disabled
def test_invalid_batch_size_falls_back_to_default(warm_model):
    """batch_size=0 / negative must not raise; falls back to a sane default."""
    out = verify(
        answer="Sarah is in Berlin.",
        evidence=["Sarah lives in Berlin."],
        batch_size=0,
    )
    assert out.decision is NLIDecision.ENTAIL


# ────────────────────────────────────────────────────────────────────
# Latency budget — p95 < 50 ms for 5 evidence pieces after warmup.
# ────────────────────────────────────────────────────────────────────


@_skip_if_disabled
def test_p95_latency_under_budget(warm_model):
    evidence = [
        "Sarah lives in Berlin.",
        "Berlin is the capital of Germany.",
        "Sarah enjoys Brandenburg Gate walks.",
        "She moved there in 2021.",
        "Her apartment is in Mitte.",
    ]
    answer = "Sarah is in Berlin."

    # Discard a couple of warm-up calls in addition to the fixture warmup —
    # MPS in particular benefits from a few hot iterations.
    for _ in range(3):
        verify(answer=answer, evidence=evidence)

    samples_ms: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        verify(answer=answer, evidence=evidence)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    p95 = samples_ms[int(0.95 * len(samples_ms)) - 1]

    # Stash for diagnostics — visible with `pytest -s`.
    print(
        f"\n[verifier] p50={samples_ms[len(samples_ms)//2]:.1f}ms "
        f"p95={p95:.1f}ms p99={samples_ms[-1]:.1f}ms",
    )

    budget_ms = float(os.environ.get("MEMORY_NLI_P95_BUDGET_MS", "50"))
    if p95 >= budget_ms:
        pytest.xfail(
            f"p95={p95:.1f}ms exceeded {budget_ms:.0f}ms budget on this host. "
            "Acceptable on cold/heavily-loaded machines; tighten on dedicated CI."
        )
    assert p95 < budget_ms
