"""Tests for deep (LLM-based) metadata enricher.

ingestion.enricher.MetadataEnricher provides fast heuristic metadata
(language, project, content_category, tokens, has_code). deep_enricher
adds LLM-extracted semantic metadata: entities, intent, topics. These are
stored alongside base metadata and used for filtering / faceted retrieval.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────


def test_parse_entities_from_json_response():
    from deep_enricher import parse_entities

    raw = '{"entities": [{"name": "Go", "type": "technology"}, {"name": "Bob", "type": "person"}]}'
    result = parse_entities(raw)
    names = {e["name"] for e in result}
    assert "Go" in names
    assert "Bob" in names


def test_parse_entities_from_markdown_fenced_json():
    from deep_enricher import parse_entities

    raw = "```json\n{\"entities\":[{\"name\":\"Redis\",\"type\":\"technology\"}]}\n```"
    result = parse_entities(raw)
    assert result[0]["name"] == "Redis"


def test_parse_entities_returns_empty_on_garbage():
    from deep_enricher import parse_entities

    assert parse_entities("not json at all") == []
    assert parse_entities("") == []


def test_parse_intent_extracts_label():
    from deep_enricher import parse_intent

    assert parse_intent('{"intent": "question"}') == "question"
    # hyphens are normalized to snake_case (see prompt contract)
    assert parse_intent("intent: how-to") == "how_to"
    assert parse_intent("   procedural   ") == "procedural"


def test_parse_intent_falls_back_to_unknown():
    from deep_enricher import parse_intent

    assert parse_intent("") == "unknown"
    assert parse_intent(None) == "unknown"  # type: ignore[arg-type]


def test_parse_topics_from_list():
    from deep_enricher import parse_topics

    raw = '["authentication", "security", "jwt"]'
    assert parse_topics(raw) == ["authentication", "security", "jwt"]


def test_parse_topics_comma_separated():
    from deep_enricher import parse_topics

    assert parse_topics("authentication, security, jwt") == ["authentication", "security", "jwt"]


def test_parse_topics_caps_count():
    from deep_enricher import parse_topics

    raw = ", ".join(f"topic{i}" for i in range(50))
    result = parse_topics(raw, max_topics=10)
    assert len(result) == 10


# ──────────────────────────────────────────────
# Full enrich pipeline
# ──────────────────────────────────────────────


def test_deep_enrich_merges_with_existing_metadata(monkeypatch):
    from deep_enricher import deep_enrich

    def fake_llm(prompt: str, **_: object) -> str:
        if "entit" in prompt.lower():
            return '{"entities": [{"name": "Go", "type": "technology"}]}'
        if "intent" in prompt.lower():
            return "procedural"
        if "topic" in prompt.lower():
            return '["backend", "microservices"]'
        return ""

    monkeypatch.setattr("deep_enricher._llm_complete", fake_llm)

    base_meta = {"language": "en", "project": "demo"}
    enriched = deep_enrich("Long document about Go microservices " * 20, base_metadata=base_meta)

    assert enriched["language"] == "en"  # preserved
    assert enriched["project"] == "demo"  # preserved
    assert "entities" in enriched
    assert "intent" in enriched
    assert "topics" in enriched
    assert enriched["intent"] == "procedural"
    assert "backend" in enriched["topics"]


def test_deep_enrich_swallows_llm_failures(monkeypatch):
    from deep_enricher import deep_enrich

    def broken(*_a: object, **_kw: object) -> str:
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr("deep_enricher._llm_complete", broken)

    result = deep_enrich("some content", base_metadata={"project": "demo"})
    # Base metadata preserved, enrichment fields default to safe values
    assert result["project"] == "demo"
    assert result.get("entities", []) == []
    assert result.get("intent") in ("unknown", None, "")
    assert result.get("topics", []) == []


def test_deep_enrich_skips_on_short_content(monkeypatch):
    """Very short content shouldn't trigger LLM calls."""
    from deep_enricher import deep_enrich

    calls: list[str] = []

    def fake(prompt: str, **_: object) -> str:
        calls.append(prompt)
        return ""

    monkeypatch.setattr("deep_enricher._llm_complete", fake)

    result = deep_enrich("hi", base_metadata={})
    assert calls == []
    # Stub values added
    assert "entities" in result
