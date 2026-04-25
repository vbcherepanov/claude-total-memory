"""v9.0 D7 — judge-weighted ensemble picker.

The default ensemble in benchmarks/locomo_bench_llm.py uses a one-shot LLM
voter that returns an index. That's cheap but noisy: voter only sees raw
strings, not the question's structure, so it picks the longest / most
confident-sounding answer regardless of correctness.

This module is a richer alternative:
  * one judge call returns a per-candidate score (0..10) AND an abstain flag,
  * score rubric is category-aware (LoCoMo cat 1..5),
  * abstain → bench returns "Not mentioned in the conversation." (matches
    LoCoMo adversarial gold style),
  * highest-score candidate wins; ties broken by earliest in the list (which
    is the temp=0 deterministic candidate by convention).

Used by --ensemble-mode=judge (added to locomo_bench_llm.py in this commit).

No coupling to the bench module — `judge_weighted_pick` takes a generic
LLMClient-like callable and returns (answer, telemetry) so it's reusable
in any harness that needs ensemble post-processing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


# Category 1..5 — same official LoCoMo mapping as locomo_bench_llm.CATEGORY_NAMES.
_CATEGORY_RUBRIC: dict[int, str] = {
    1: (
        "Single-hop factual: answer should be 1-6 words, copying or paraphrasing a "
        "specific phrase from the conversation. Reward terse, exact noun-phrase "
        "answers; penalize verbose explanations."
    ),
    2: (
        "Temporal: answer must be a date/duration/time expression (\"7 May 2023\", "
        "\"two weeks ago\", \"4 years\"). Reward ISO-or-near-ISO dates anchored to "
        "the conversation; penalize vague answers."
    ),
    3: (
        "Multi-hop: answer must combine 2+ facts. A short reasoned answer "
        "(\"Likely yes, because she enjoys hiking and national parks fit that\") "
        "is correct. Penalize answers that ignore one of the entities mentioned "
        "in the question."
    ),
    4: (
        "Open-domain: answer summarizes a person's traits, life, or preferences. "
        "Reward 1-2 short sentence answers covering main attributes; penalize "
        "list dumps or speculative additions."
    ),
    5: (
        "Adversarial / unanswerable: if the topic is absent from the conversation, "
        "the correct answer is a refusal (\"Not mentioned in the conversation.\"). "
        "Reward refusal when context is thin; penalize confident hallucinations."
    ),
}

_DEFAULT_RUBRIC = (
    "Reward concise answers grounded in conversation context. Penalize "
    "hallucinations, fabricated entities, or speculation beyond the excerpts."
)


JUDGE_SYSTEM = """You evaluate candidate answers to a question about a conversation.
For each candidate, score how well it answers the question on a 0-10 scale.

Output STRICT JSON on a single line:
{"scores": [s0, s1, ...], "abstain": bool, "reason": "..."}

Where:
- scores: float between 0 and 10 per candidate, in input order.
- abstain: true if all candidates are equally weak / all hallucinate / context is insufficient.
- reason: one short phrase explaining the top pick.

No prose, no markdown fences."""


@dataclass
class JudgePick:
    answer: str
    scores: list[float]
    abstain: bool
    reason: str
    judge_input_tokens: int
    judge_output_tokens: int


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _build_user_prompt(question: str, candidates: list[str], category: int | None) -> str:
    rubric = _CATEGORY_RUBRIC.get(category or 0, _DEFAULT_RUBRIC)
    body = [
        f"CATEGORY_RUBRIC: {rubric}",
        f"QUESTION: {question}",
        "",
        "CANDIDATES:",
    ]
    for i, c in enumerate(candidates):
        body.append(f"[{i}] {c}")
    body.append("")
    body.append("Score each candidate (0-10), set abstain=true if all are weak, then output JSON.")
    return "\n".join(body)


def _parse_response(
    raw_text: str, n_candidates: int
) -> tuple[list[float], bool, str]:
    """Returns (scores, abstain, reason). Falls back to neutral on parse fail."""
    text = _strip_fences(raw_text)
    try:
        obj = json.loads(text)
    except Exception:
        return [5.0] * n_candidates, False, "parse_error"

    raw_scores = obj.get("scores")
    scores: list[float] = []
    if isinstance(raw_scores, list):
        for s in raw_scores:
            try:
                scores.append(max(0.0, min(10.0, float(s))))
            except (TypeError, ValueError):
                scores.append(5.0)

    if len(scores) < n_candidates:
        scores += [5.0] * (n_candidates - len(scores))
    scores = scores[:n_candidates]

    abstain = bool(obj.get("abstain", False))
    reason = str(obj.get("reason", "")).strip() or "no_reason"
    return scores, abstain, reason


def judge_weighted_pick(
    client,
    *,
    question: str,
    candidates: list[str],
    category: int | None,
    judge_model: str,
    abstain_answer: str = "Not mentioned in the conversation.",
    min_score_floor: float = 3.0,
) -> JudgePick:
    """Single judge call → score per candidate → pick highest, with abstain logic.

    Args:
        client: object with `.complete(system, user, *, model, max_tokens)`
            returning an object with .text/.input_tokens/.output_tokens (matches
            benchmarks._llm_adapter.LLMClient).
        question: the question being answered.
        candidates: list of candidate answer strings (typically k=2..4).
        category: LoCoMo category 1..5, or None for the default rubric.
        judge_model: model id for the judge call.
        abstain_answer: returned when judge flags all candidates as weak OR
            the max score is below `min_score_floor`. Default matches LoCoMo
            adversarial gold style.
        min_score_floor: numeric threshold below which we treat the result
            as an abstain even if the judge didn't set the flag.

    Returns:
        :class:`JudgePick` with the chosen answer + scoring telemetry.
    """
    if not candidates:
        return JudgePick(
            answer=abstain_answer,
            scores=[],
            abstain=True,
            reason="no_candidates",
            judge_input_tokens=0,
            judge_output_tokens=0,
        )

    if len(candidates) == 1:
        # Nothing to weigh — return as-is.
        return JudgePick(
            answer=candidates[0],
            scores=[10.0],
            abstain=False,
            reason="single_candidate",
            judge_input_tokens=0,
            judge_output_tokens=0,
        )

    user = _build_user_prompt(question, candidates, category)
    try:
        r = client.complete(JUDGE_SYSTEM, user, model=judge_model, max_tokens=120)
        raw_text = getattr(r, "text", "") or ""
        in_tok = int(getattr(r, "input_tokens", 0) or 0)
        out_tok = int(getattr(r, "output_tokens", 0) or 0)
    except Exception:
        # Hard-fail safety: pick the first candidate, no telemetry.
        return JudgePick(
            answer=candidates[0],
            scores=[5.0] * len(candidates),
            abstain=False,
            reason="judge_call_failed",
            judge_input_tokens=0,
            judge_output_tokens=0,
        )

    scores, abstain_flag, reason = _parse_response(raw_text, len(candidates))
    max_score = max(scores) if scores else 0.0

    if abstain_flag or max_score < min_score_floor:
        return JudgePick(
            answer=abstain_answer,
            scores=scores,
            abstain=True,
            reason=reason or "below_floor",
            judge_input_tokens=in_tok,
            judge_output_tokens=out_tok,
        )

    # Tie-break by earliest (temp=0 candidate).
    best_idx = 0
    for i, s in enumerate(scores):
        if s > scores[best_idx]:
            best_idx = i

    return JudgePick(
        answer=candidates[best_idx],
        scores=scores,
        abstain=False,
        reason=reason,
        judge_input_tokens=in_tok,
        judge_output_tokens=out_tok,
    )
