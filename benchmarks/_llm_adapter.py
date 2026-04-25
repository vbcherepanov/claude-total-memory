"""Thin LLM provider adapter used by LoCoMo bench + fact_synthesizer.

Goal: same call-site regardless of backend, so we can flip between Claude
(anthropic.Anthropic) and OpenAI (gpt-4o / gpt-4o-mini / o-series) with a
single --provider flag. No other changes needed in the bench loop.

Deliberately tiny — token accounting normalised, retry/backoff uniform,
system prompt handled both as Anthropic ``system=`` and OpenAI
``messages[0].role="system"``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


# Canonical aliases for both providers. CLI accepts short names.
MODEL_ALIASES: dict[str, str] = {
    # Anthropic
    "haiku": "claude-haiku-4-5-20251001",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
    "opus-4.7": "claude-opus-4-7",
    # OpenAI
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini",
    "o1-mini": "o1-mini",
}


# Which provider owns which model prefix — used to auto-resolve --provider=auto.
def detect_provider(model: str) -> str:
    m = (model or "").lower()
    if m.startswith(("claude", "haiku", "sonnet", "opus")):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "4o")):
        return "openai"
    # Fallback: whichever env key is present.
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    """Shared provider interface — .complete(system, user, model, max_tokens)."""

    def __init__(self, provider: str = "auto", default_model: str | None = None) -> None:
        self.provider = provider if provider != "auto" else detect_provider(default_model or "")
        self._anthropic: Any = None
        self._openai: Any = None
        if self.provider == "anthropic":
            import anthropic  # noqa: PLC0415
            self._anthropic = anthropic.Anthropic()
        elif self.provider == "openai":
            from openai import OpenAI  # noqa: PLC0415
            # OpenAI() reads OPENAI_API_KEY from env.
            self._openai = OpenAI()
        else:
            raise ValueError(f"unsupported provider: {self.provider!r}")

    # ── public ──────────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retries: int = 3,
    ) -> LLMResult:
        """Run one completion. On failure, exponential backoff."""
        model = MODEL_ALIASES.get(model, model)
        last: Exception | None = None
        for attempt in range(retries):
            try:
                if self.provider == "anthropic":
                    return self._call_anthropic(system, user, model, max_tokens, temperature)
                return self._call_openai(system, user, model, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (2 ** attempt))
        raise RuntimeError(f"{self.provider} call failed after {retries} retries: {last}")

    # ── per-provider implementation ─────────────────────────────────────

    def _call_anthropic(
        self, system: str, user: str, model: str, max_tokens: int, temperature: float
    ) -> LLMResult:
        r = self._anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = r.content[0].text.strip() if r.content else ""
        return LLMResult(text, r.usage.input_tokens, r.usage.output_tokens)

    def _call_openai(
        self, system: str, user: str, model: str, max_tokens: int, temperature: float
    ) -> LLMResult:
        # The o1-* family doesn't accept system messages / temperature yet;
        # fold the system into the user turn for those.
        is_o_series = model.startswith("o1") or model.startswith("o3")
        if is_o_series:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            kwargs: dict = {"model": model, "messages": messages,
                            "max_completion_tokens": max_tokens}
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        r = self._openai.chat.completions.create(**kwargs)
        text = (r.choices[0].message.content or "").strip()
        usage = r.usage
        return LLMResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


__all__ = ["LLMClient", "LLMResult", "MODEL_ALIASES", "detect_provider"]
