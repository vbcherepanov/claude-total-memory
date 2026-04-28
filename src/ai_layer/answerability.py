"""Answerability classifier (W1-D, v11) — gate before generation.

LoCoMo "adversarial" questions have no correct answer in the supplied
memory; the LLM is supposed to say "I don't know". Today our pipeline
hallucinates because nothing checks "does the evidence actually support
an answer?" before we hand it to the writer.

This module is that check: one Haiku call, strict JSON, evidence-only
reasoning. The result is consumed by ``memory_core.idk_router.route``
to decide between ANSWER / ANSWER_WITH_CAVEAT / SEARCH_MORE / IDK.

Design notes
------------

* **Single LLM call.** The call is light (small system + small user)
  but every retrieval round pays for it, so we keep ``max_tokens``
  tight and temperature at 0.

* **Up to 2 retries on parse error.** The model is asked for one
  one-line JSON object; if it returns prose or broken JSON we retry
  with the same prompt. After the 3rd failure we fall through to a
  deterministic "not answerable" verdict so the caller never blocks.

* **No world knowledge.** The system prompt forbids the model from
  using anything outside the evidence list. This is the whole point —
  if the evidence does not contain the answer we want IDK, not a
  plausible guess.

* **Empty evidence shortcut.** ``evidence == []`` skips the LLM call
  and returns a hard "not answerable" with confidence 1.0. Saves a
  round-trip and removes one class of LLM-noise failures.

The module deliberately exposes ``llm_client`` as an injection point so
unit tests can pass a fake without monkeypatching ``_llm_adapter``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


# Resolved model id for tests/operators that want the canonical name —
# matches the alias in benchmarks/_llm_adapter.MODEL_ALIASES.
DEFAULT_MODEL = "haiku"

# Cap on evidence chars per snippet sent to the LLM. Real LoCoMo
# snippets are short paragraphs, but pathological 50KB blobs would
# blow up the prompt; we truncate per-snippet rather than per-prompt
# so each snippet still contributes.
_MAX_SNIPPET_CHARS = 800
_MAX_TOTAL_EVIDENCE_CHARS = 6000
_MAX_RETRIES = 2  # ie. up to 3 attempts including the initial call
_MAX_OUTPUT_TOKENS = 200


# ──────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────


@dataclass
class AnswerabilityResult:
    """Verdict on whether ``evidence`` supports answering ``question``."""

    answerable: bool          # evidence fully supports an answer
    partial: bool             # some support, but missing pieces
    confidence: float         # 0..1
    missing: str | None       # one short clause describing what is absent
    rationale: str            # one sentence explanation


class _LLMLike(Protocol):
    """Minimal protocol the classifier needs from a client.

    Matches ``benchmarks._llm_adapter.LLMClient.complete`` — anything
    callable with the same kwargs and returning either a string or an
    object with a ``.text`` attribute is acceptable. The protocol stays
    duck-typed so test fakes do not have to subclass anything.
    """

    def complete(self, **kwargs: Any) -> Any:  # pragma: no cover - protocol
        ...


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────


_SYSTEM_PROMPT = (
    "You are an answerability classifier for a memory-augmented assistant. "
    "Decide whether the supplied EVIDENCE fully supports a direct answer to "
    "the QUESTION. You MUST NOT use world knowledge, common sense facts, or "
    "anything outside the evidence. If the evidence is empty, off-topic, or "
    "only tangentially related, the question is NOT answerable.\n\n"
    "Output a SINGLE JSON object on ONE line, no prose, no markdown fences:\n"
    "{\"answerable\": bool, \"partial\": bool, \"confidence\": 0.0-1.0, "
    "\"missing\": string|null, \"rationale\": string}\n\n"
    "Definitions:\n"
    "- answerable=true: every fact needed to answer is explicitly in the "
    "evidence. partial must be false.\n"
    "- partial=true: evidence touches the topic but a key piece is missing "
    "(date, name, value, etc). answerable must be false.\n"
    "- both false: evidence is irrelevant or contradicts the question.\n"
    "- confidence: how sure you are about the verdict, NOT the answer.\n"
    "- missing: one short clause naming the absent fact, or null when "
    "answerable=true.\n"
    "- rationale: one sentence, <= 200 chars."
)


def _build_user_prompt(question: str, evidence: list[str]) -> str:
    pieces: list[str] = ["QUESTION:", question.strip(), "", "EVIDENCE:"]
    total = 0
    for i, snippet in enumerate(evidence, 1):
        text = (snippet or "").strip()
        if not text:
            continue
        if len(text) > _MAX_SNIPPET_CHARS:
            text = text[:_MAX_SNIPPET_CHARS] + "…"
        budget_left = _MAX_TOTAL_EVIDENCE_CHARS - total
        if budget_left <= 0:
            pieces.append(f"[{i}] …[truncated, evidence cap reached]")
            break
        if len(text) > budget_left:
            text = text[:budget_left] + "…"
        pieces.append(f"[{i}] {text}")
        total += len(text)
    pieces.append("")
    pieces.append("Respond with JSON only.")
    return "\n".join(pieces)


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────


def _extract_text(raw: Any) -> str:
    """Normalise an LLM response into a string.

    The shared adapter returns ``LLMResult(text=...)`` but several test
    fakes — including the one in our own test file — return a bare
    string. Both must work.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    text_attr = getattr(raw, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(raw)


def _find_json_object(text: str) -> str | None:
    """Locate the first balanced JSON object in ``text``.

    The model is instructed to emit one JSON object only, but some
    responses still wrap the object in markdown fences or stray prose.
    A balanced-brace scan handles nested objects in ``rationale`` if
    they ever appear, where a regex would not.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _parse_response(raw: Any) -> AnswerabilityResult | None:
    """Parse one LLM response. ``None`` signals a retry-worthy failure."""
    text = _extract_text(raw).strip()
    if not text:
        return None
    # Strip common markdown fences first.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    blob = _find_json_object(text)
    if blob is None:
        return None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    answerable = obj.get("answerable")
    partial = obj.get("partial")
    confidence = obj.get("confidence")
    missing = obj.get("missing")
    rationale = obj.get("rationale")

    if not isinstance(answerable, bool):
        return None
    if not isinstance(partial, bool):
        return None
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return None
    conf = max(0.0, min(1.0, conf))

    # Schema invariant: answerable and partial cannot both be true.
    # When the model violates it, prefer the safer "partial" reading —
    # a caveat is less harmful than a confident wrong answer.
    if answerable and partial:
        answerable = False

    if missing is not None and not isinstance(missing, str):
        missing = str(missing)
    if missing is not None:
        missing = missing.strip()[:240] or None
    if isinstance(rationale, str):
        rationale_text = rationale.strip()[:280]
    else:
        rationale_text = ""
    if not rationale_text:
        rationale_text = "no rationale provided"

    if answerable and missing:
        # Inconsistent — model said yes but listed a missing piece.
        # Treat it as partial so downstream caveats show the gap.
        answerable = False
        partial = True

    return AnswerabilityResult(
        answerable=answerable,
        partial=partial,
        confidence=conf,
        missing=missing,
        rationale=rationale_text,
    )


# ──────────────────────────────────────────────
# Default LLM client
# ──────────────────────────────────────────────


def _default_llm_client() -> _LLMLike:
    """Build the shared adapter once per process.

    Importing the bench adapter eagerly would force ``anthropic`` /
    ``openai`` into every test that touches answerability, even when a
    fake client is injected. We import lazily so unit tests stay
    offline.
    """
    from benchmarks._llm_adapter import LLMClient  # noqa: PLC0415
    return LLMClient(provider="auto", default_model=DEFAULT_MODEL)


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────


def classify_answerability(
    question: str,
    evidence: list[str],
    *,
    llm_model: str = DEFAULT_MODEL,
    llm_client: _LLMLike | None = None,
) -> AnswerabilityResult:
    """Classify whether ``evidence`` supports answering ``question``.

    Parameters
    ----------
    question:
        Natural-language user question. Trimmed to a sane size for the
        prompt; empty/whitespace input is treated as a hard "no".
    evidence:
        Retrieved snippets, in any order. Empty list short-circuits
        without an LLM call.
    llm_model:
        Model alias passed to the client (default: Haiku, see
        ``benchmarks._llm_adapter.MODEL_ALIASES``).
    llm_client:
        Optional dependency-injection hook. Anything with a
        ``.complete(system=, user=, model=, max_tokens=, temperature=)``
        method works. When omitted we lazily build the shared adapter.

    Returns
    -------
    AnswerabilityResult
        Always returns a valid result. On total LLM/parse failure we
        return a conservative "not answerable" verdict so callers can
        route to IDK without crashing.
    """
    q = (question or "").strip()
    cleaned_evidence = [e for e in (evidence or []) if (e or "").strip()]

    if not q:
        return AnswerabilityResult(
            answerable=False,
            partial=False,
            confidence=1.0,
            missing="empty question",
            rationale="empty question — nothing to classify",
        )
    if not cleaned_evidence:
        return AnswerabilityResult(
            answerable=False,
            partial=False,
            confidence=1.0,
            missing="no evidence",
            rationale="no evidence supplied; cannot ground an answer",
        )

    client: _LLMLike
    if llm_client is None:
        try:
            client = _default_llm_client()
        except Exception as exc:
            return AnswerabilityResult(
                answerable=False,
                partial=False,
                confidence=0.5,
                missing="llm unavailable",
                rationale=f"llm client unavailable ({type(exc).__name__})",
            )
    else:
        client = llm_client

    user_prompt = _build_user_prompt(q, cleaned_evidence)

    last_error: str = ""
    attempts = _MAX_RETRIES + 1
    for attempt in range(attempts):
        try:
            raw = client.complete(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                model=llm_model,
                max_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.0,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue

        parsed = _parse_response(raw)
        if parsed is not None:
            return parsed
        last_error = "unparsable response"

    return AnswerabilityResult(
        answerable=False,
        partial=False,
        confidence=0.5,
        missing="classifier unavailable",
        rationale=f"classifier failed after {attempts} attempts: {last_error}",
    )


__all__ = [
    "AnswerabilityResult",
    "DEFAULT_MODEL",
    "classify_answerability",
]
