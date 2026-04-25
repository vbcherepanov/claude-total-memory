"""v9.0 D3 — schema-specific predicate extraction (canonical vocabulary)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import extract_triples_openai as et  # noqa: E402


class _FakeClient:
    def __init__(self, response_text: str):
        self.calls: list[dict] = []
        self._response_text = response_text
        self.provider = "openai"

    def complete(self, system: str, user: str, *, model: str, max_tokens: int):
        self.calls.append({"system": system, "user": user, "model": model})
        return SimpleNamespace(text=self._response_text, input_tokens=10, output_tokens=20)


def test_v2_accepts_canonical_predicates():
    payload = {
        "triples": [
            {"s": "Caroline", "r": "person_went_to", "o": "Paris"},
            {"s": "Anna", "r": "person_age", "o": "5"},
        ]
    }
    client = _FakeClient(json.dumps(payload))
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v2")
    assert len(triples) == 2
    assert triples[0]["r"] == "person_went_to"
    assert triples[1]["r"] == "person_age"


def test_v2_rejects_synonyms():
    """Free-form 'traveled_to' should be dropped — LLM must collapse to person_went_to."""
    payload = {
        "triples": [
            {"s": "X", "r": "traveled_to", "o": "Paris"},          # synonym → drop
            {"s": "X", "r": "person_went_to", "o": "Berlin"},      # canonical → keep
            {"s": "X", "r": "had_dinner_with", "o": "Y"},          # not in vocab → drop
        ]
    }
    client = _FakeClient(json.dumps(payload))
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v2")
    assert [t["r"] for t in triples] == ["person_went_to"]


def test_v2_accepts_other_escape_hatch():
    payload = {
        "triples": [
            {"s": "Bob", "r": "other:plays_instrument", "o": "violin"},
        ]
    }
    client = _FakeClient(json.dumps(payload))
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v2")
    assert len(triples) == 1
    assert triples[0]["r"] == "other:plays_instrument"


def test_v2_rejects_bare_other():
    """Bare 'other' is meaningless — must carry suffix `:<freeform>`."""
    payload = {"triples": [{"s": "X", "r": "other", "o": "Y"}]}
    client = _FakeClient(json.dumps(payload))
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v2")
    assert triples == []


def test_v1_still_accepts_freeform():
    payload = {
        "triples": [
            {"s": "X", "r": "traveled_to", "o": "Paris"},
            {"s": "Y", "r": "had_dinner_with", "o": "Z"},
        ]
    }
    client = _FakeClient(json.dumps(payload))
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v1")
    assert len(triples) == 2


def test_extract_drops_oversized_strings():
    payload = {
        "triples": [
            {"s": "x" * 200, "r": "person_went_to", "o": "Paris"},
            {"s": "Z", "r": "person_went_to", "o": "y" * 250},
            {"s": "ok", "r": "person_went_to", "o": "Paris"},
        ]
    }
    client = _FakeClient(json.dumps(payload))
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v2")
    assert len(triples) == 1
    assert triples[0]["s"] == "ok"


def test_extract_handles_invalid_json():
    client = _FakeClient("not json at all")
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v2")
    assert triples == []


def test_extract_handles_fences():
    payload = {"triples": [{"s": "X", "r": "person_went_to", "o": "Y"}]}
    client = _FakeClient("```json\n" + json.dumps(payload) + "\n```")
    triples = et.extract(client, "x", model="gpt-4o-mini", prompt_version="v2")
    assert len(triples) == 1


def test_canonical_predicates_v2_includes_required_set():
    required = {
        "person_went_to", "person_age", "person_relationship",
        "person_bought", "person_paid", "event_occurred_on",
        "event_location", "person_has_pet",
    }
    assert required <= set(et.CANONICAL_PREDICATES_V2)
    # "other" is intentionally NOT in CANONICAL_PREDICATES_V2 — it's a
    # suffix-only escape hatch enforced by _normalize_predicate.
    assert "other" not in set(et.CANONICAL_PREDICATES_V2)


def test_default_system_prompt_is_v2():
    assert et.SYSTEM_PROMPT is et.SYSTEM_PROMPT_V2


def test_normalize_predicate_lowercases_and_underscores():
    assert et._normalize_predicate("Person_Went_To", "v2") == "person_went_to"
    assert et._normalize_predicate("had dinner with", "v1") == "had_dinner_with"


def test_normalize_predicate_drops_long_freeform():
    too_long = "other:" + "x" * 70
    assert et._normalize_predicate(too_long, "v2") is None
