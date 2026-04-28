"""v11 answer-pipeline post-processor.

Sits AFTER the existing locomo_bench_llm.py answer generator. Takes a
predicted answer + retrieved evidence + the QA category, runs:

  1. NLI verification (W1-E)            — does evidence entail the answer?
  2. Answerability classification (W1-D)— would a permissive judge accept it?
  3. Calibrated routing (W2-I)          — ANSWER / CAVEAT / HYBRID / IDK
  4. Optional negative retrieval (W2-H) — adversarial questions only

If the router emits IDK, the prediction is rewritten to a LoCoMo-style
refusal phrase. Otherwise the original prediction is kept (we never
fabricate a different answer).

Why post-processing only?
  - Our failure analysis shows 30% of errors are over-cautious refusals.
    A pre-gate would hurt single-hop where evidence IS retrieved.
  - The base pipeline already implements ensemble/qrw/hyde/per-cat-prompts
    well (the v9 archive's best run, ensemble3, hit 0.696). Re-doing
    that work risks regression. v11 = additive, not replacement.

Public API:
    apply_v11_pipeline(client, question, gold, pred, entries, *, category,
                       gen_model, judge_model, enable_verifier, enable_router,
                       enable_negative, llm_client_for_planner) -> dict
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


CATEGORY_LABEL = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

REFUSAL_TEXT = "Not specified in the conversation."


def _avg_retrieval_score(entries: list[dict]) -> float:
    """Mean normalised retrieval score over top-K. Falls back to 0.5 if
    the bench-supplied entries don't carry a score field."""
    if not entries:
        return 0.0
    scores: list[float] = []
    for e in entries[:10]:
        s = e.get("score") or e.get("rrf_score") or e.get("rank_score")
        if s is None:
            continue
        try:
            scores.append(float(s))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.5
    # Normalise into [0, 1]: existing bench scores can exceed 1 (RRF additive).
    mx = max(scores)
    if mx <= 0:
        return 0.0
    if mx <= 1.0:
        return sum(scores) / len(scores)
    return min(1.0, (sum(scores) / len(scores)) / mx)


def _build_evidence_list(entries: list[dict], cap: int = 10) -> list[str]:
    out: list[str] = []
    for e in entries[:cap]:
        text = e.get("text") or e.get("content") or e.get("dialogue") or ""
        if isinstance(text, str) and text.strip():
            out.append(text.strip()[:600])
    return out


def apply_v11_pipeline(
    client,
    question: str,
    pred: str,
    entries: list[dict],
    *,
    category: int = 0,
    gen_model: str = "gpt-4o",
    judge_model: str = "haiku",
    enable_verifier: bool = True,
    enable_router: bool = True,
    enable_negative: bool = False,
    contradiction_fn=None,
) -> dict:
    """Returns:
        {
          "final_pred": str,           # possibly rewritten if router → IDK
          "original_pred": str,
          "route": str,                # answer | caveat | hybrid | search_more | idk
          "answerability": dict,
          "nli": dict,
          "negative": dict | None,
          "tokens_in": int,
          "tokens_out": int,
        }
    """
    cat_label = CATEGORY_LABEL.get(category)
    evidence_texts = _build_evidence_list(entries)
    raw_score = _avg_retrieval_score(entries)
    tokens_in = 0
    tokens_out = 0
    out: dict[str, Any] = {
        "final_pred": pred,
        "original_pred": pred,
        "route": "answer",
        "answerability": {},
        "nli": {},
        "negative": None,
        "tokens_in": 0,
        "tokens_out": 0,
    }

    # 1. NLI verification (local, ~12 ms p95)
    nli = {"decision": "neutral", "p_entail": 0.0, "p_neutral": 1.0, "p_contradict": 0.0}
    if enable_verifier and evidence_texts and pred and pred.strip() and pred != REFUSAL_TEXT:
        try:
            from ai_layer.verifier import verify
            vr = verify(pred, evidence_texts)
            nli = {
                "decision": vr.decision.value if hasattr(vr.decision, "value") else str(vr.decision),
                "p_entail": float(vr.p_entail),
                "p_neutral": float(vr.p_neutral),
                "p_contradict": float(vr.p_contradict),
            }
        except Exception as e:
            nli["error"] = str(e)
    out["nli"] = nli

    # 2. Answerability — uses Anthropic Haiku via existing _llm_adapter.
    answerability = {"answerable": True, "partial": False, "confidence": 0.6}
    if enable_router and evidence_texts:
        try:
            from ai_layer.answerability import classify_answerability

            class _Adapter:
                def __init__(self, c):
                    self._c = c

                def complete(self, *, model, system, user, max_tokens):
                    r = self._c.complete(system, user, model=model, max_tokens=max_tokens, temperature=0.0)
                    return r.text

            ans_client = _Adapter(client) if client is not None else None
            ar = classify_answerability(
                question, evidence_texts,
                llm_model=judge_model,
                llm_client=ans_client,
            )
            answerability = {
                "answerable": bool(ar.answerable),
                "partial": bool(ar.partial),
                "confidence": float(ar.confidence),
                "missing": ar.missing,
                "rationale": ar.rationale,
            }
        except Exception as e:
            answerability["error"] = str(e)
    out["answerability"] = answerability

    # 3. Negative retrieval — adversarial only, opt-in.
    has_contradiction = False
    if enable_negative and category == 5 and entries and contradiction_fn is not None:
        try:
            from memory_core.negative_retrieval import negative_retrieve

            def _search(q, k=10, project=None):
                # No additional retrieval here — adversarial flow uses positive
                # entries cross-checked against contradiction_fn directly.
                return entries[:k]

            class _LLMAdapter:
                def __init__(self, c):
                    self._c = c

                def complete(self, *, model, system, user, max_tokens):
                    r = self._c.complete(system, user, model=model, max_tokens=max_tokens, temperature=0.0)
                    return r.text

            nr = negative_retrieve(
                question, entries,
                search_fn=_search,
                contradiction_fn=contradiction_fn,
                llm_model=judge_model,
                llm_client=_LLMAdapter(client) if client is not None else None,
            )
            has_contradiction = nr.decision == "hard_contradict"
            out["negative"] = {
                "decision": nr.decision,
                "score": nr.contradiction_score,
                "rationale": nr.rationale,
            }
        except Exception as e:
            out["negative"] = {"error": str(e)}

    # 4. Routing decision.
    route_action = "answer"
    if enable_router:
        try:
            from memory_core.answer_router import (
                RouteAction, RoutingInputs, route as _route,
            )
            # NOTE: this pipeline runs after the bench has already produced
            # a candidate `pred`. There is no IRCoT loop here — the router
            # must treat us as the *final* iteration so it doesn't park on
            # SEARCH_MORE and pass the original pred through unchanged.
            # iters_done == max_iters forces the router to choose between
            # ANSWER / CAVEAT / HYBRID / IDK on the evidence we already have.
            inputs = RoutingInputs(
                category=cat_label,
                raw_retrieval_score=raw_score,
                answerable=answerability.get("answerable", True),
                partial_answerable=answerability.get("partial", False),
                answerability_confidence=answerability.get("confidence", 0.6),
                nli_decision=nli.get("decision", "neutral"),
                nli_p_contradict=nli.get("p_contradict", 0.0),
                iters_done=4,
                max_iters=4,
                has_contradiction=has_contradiction,
            )
            decision = _route(inputs)
            route_action = decision.action.value if hasattr(decision.action, "value") else str(decision.action)
            out["route_reason"] = decision.reason
            out["calibrated_p"] = decision.calibrated_p
            out["threshold_used"] = decision.threshold_used
        except Exception as e:
            out["route_error"] = str(e)

    out["route"] = route_action

    # 5. Apply route to final answer.
    #
    # Lesson from the Haiku 200-QA A/B (2026-04-28): trusting the router to
    # *override* the generator's `pred` was a net loss — when the router
    # forced IDK on `answerable=False`, single-hop dropped from 0.13 to 0.0
    # because the answerability classifier under-counts evidence the
    # generator was already exploiting. The base pipeline (ensemble3 in v9)
    # is calibrated; v11 must add signal, not subtract it.
    #
    # New policy — v11 only overrides `pred` in two cases where it has
    # high-precision veto signal:
    #   (a) NLI contradict with p_contradict > 0.5 — the generator
    #       confidently said something the evidence flatly disagrees with.
    #   (b) negative_retrieve returned hard_contradict — a contradicting
    #       fact was actively found in memory.
    # Otherwise we keep the generator's prediction. Routing decisions are
    # still logged so we can analyse them post-hoc.
    nli_says_contradict = (
        nli.get("decision") == "contradict"
        and nli.get("p_contradict", 0.0) > 0.5
    )
    if nli_says_contradict or has_contradiction:
        out["final_pred"] = REFUSAL_TEXT
        out["v11_overrode"] = True
    else:
        out["final_pred"] = pred
        out["v11_overrode"] = False

    out["tokens_in"] = tokens_in
    out["tokens_out"] = tokens_out
    return out


def warmup() -> None:
    """Pre-load NLI model so the first QA in the bench doesn't pay it."""
    if os.environ.get("V11_SKIP_NLI", "0") == "1":
        return
    try:
        from ai_layer.verifier import warmup as _warm
        _warm()
    except Exception:
        pass


__all__ = ["apply_v11_pipeline", "warmup", "REFUSAL_TEXT"]
