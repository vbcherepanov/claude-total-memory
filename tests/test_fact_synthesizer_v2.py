"""v9.0 D2 — LoCoMo-style fact synthesizer prompt routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import fact_synthesizer as fs  # noqa: E402


class _FakeClient:
    """Mimics benchmarks._llm_adapter.LLMClient.complete shape."""

    def __init__(self, response_text: str):
        self.calls: list[dict] = []
        self._response_text = response_text

    def complete(self, system: str, user: str, *, model: str, max_tokens: int):
        self.calls.append(
            {"system": system, "user": user, "model": model, "max_tokens": max_tokens}
        )
        return SimpleNamespace(text=self._response_text, input_tokens=10, output_tokens=20)


def test_prompt_v2_used_by_default():
    client = _FakeClient(json.dumps({"facts": ["Alice went to Paris on May 7, 2023."]}))
    facts, tin, tout = fs.synthesize_facts(
        client, "Alice: I went to Paris last weekend.",
        model="gpt-4o", turn_date="2023-05-08",
    )
    assert facts == ["Alice went to Paris on May 7, 2023."]
    assert tin == 10 and tout == 20

    call = client.calls[0]
    assert "TURN_DATE: 2023-05-08" in call["user"]
    assert "TURN: Alice: I went to Paris" in call["user"]
    # v2 system prompt carries the few-shot examples
    assert "Few-shot examples" in call["system"]
    assert "LGBTQ support group" in call["system"]


def test_prompt_v1_keeps_legacy_user_format():
    client = _FakeClient(json.dumps({"facts": ["x."]}))
    fs.synthesize_facts(
        client, "Some content",
        model="gpt-4o-mini", turn_date="2023-01-01", prompt_version="v1",
    )
    call = client.calls[0]
    # v1 format uses "Turn content:" and does NOT inject TURN_DATE/few-shots
    assert call["user"].startswith("Turn content:")
    assert "TURN_DATE" not in call["user"]
    assert "Few-shot examples" not in call["system"]


def test_prompt_versions_dict_exposes_both():
    assert "v1" in fs.PROMPT_VERSIONS
    assert "v2" in fs.PROMPT_VERSIONS
    assert fs.PROMPT_VERSIONS["v2"] is fs.SYSTEM_PROMPT_V2
    assert fs.PROMPT_VERSIONS["v1"] is fs.SYSTEM_PROMPT_V1
    # Default global alias points at v2
    assert fs.SYSTEM_PROMPT is fs.SYSTEM_PROMPT_V2


def test_unknown_prompt_version_falls_back_to_default():
    client = _FakeClient(json.dumps({"facts": []}))
    fs.synthesize_facts(client, "x", model="gpt-4o", prompt_version="unknown")
    # Should not raise; system prompt still set
    assert client.calls[0]["system"] == fs.SYSTEM_PROMPT


def test_synthesize_facts_handles_malformed_json():
    client = _FakeClient("not json at all")
    facts, tin, tout = fs.synthesize_facts(client, "x", model="gpt-4o")
    assert facts == []
    assert tin == 10 and tout == 20


def test_synthesize_facts_strips_fences():
    client = _FakeClient("```json\n" + json.dumps({"facts": ["fenced fact."]}) + "\n```")
    facts, _, _ = fs.synthesize_facts(client, "x", model="gpt-4o")
    assert facts == ["fenced fact."]


def test_synthesize_facts_filters_non_string_entries():
    client = _FakeClient(json.dumps({"facts": ["good.", 42, "", "  also good."]}))
    facts, _, _ = fs.synthesize_facts(client, "x", model="gpt-4o")
    assert facts == ["good.", "also good."]
