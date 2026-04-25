"""Integration wiring: ConceptExtractor / deep_enricher / representations /
reflection route through the `llm_provider` abstraction based on env.

Every HTTP hop is mocked via urllib.request.urlopen — no real network.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import urllib.error
from pathlib import Path

import pytest

# Make src/ importable (conftest already does this, keep explicit for safety).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


def _capture_urlopen(payload: dict, sink: dict):
    """Return a fake urlopen that records the outgoing request."""

    def fake(req, timeout=None, context=None, **_kwargs):
        sink["url"] = req.full_url
        sink["headers"] = dict(req.headers)
        sink["body"] = json.loads(req.data.decode("utf-8")) if req.data else None
        sink["timeout"] = timeout
        sink["ssl_context"] = context
        return _FakeResp(payload)

    return fake


def _reset_provider_caches() -> None:
    """Clear module-level provider caches so env changes take effect."""
    for mod_name in ("ingestion.extractor", "deep_enricher", "representations"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_provider_cache"):
            mod._provider_cache.clear()


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Ensure each test starts from a known provider-config baseline."""
    for var in (
        "MEMORY_LLM_PROVIDER",
        "MEMORY_TRIPLE_PROVIDER",
        "MEMORY_ENRICH_PROVIDER",
        "MEMORY_REPR_PROVIDER",
        "MEMORY_LLM_MODEL",
        "MEMORY_TRIPLE_MODEL",
        "MEMORY_ENRICH_MODEL",
        "MEMORY_REPR_MODEL",
        "MEMORY_LLM_API_KEY",
        "MEMORY_LLM_API_BASE",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    # Force the LLM gate open so has_llm() doesn't short-circuit.
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "force")
    _reset_provider_caches()
    yield
    _reset_provider_caches()


# ──────────────────────────────────────────────
# ConceptExtractor
# ──────────────────────────────────────────────


def test_extractor_uses_configured_provider(monkeypatch):
    """MEMORY_TRIPLE_PROVIDER=openai → HTTP goes to OpenAI with Bearer auth."""
    monkeypatch.setenv("MEMORY_TRIPLE_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_LLM_API_KEY", "sk-test-openai")
    monkeypatch.setenv("MEMORY_TRIPLE_MODEL", "gpt-4o-mini")

    import llm_provider

    sink: dict = {}
    fake = _capture_urlopen(
        {"choices": [{"message": {"content": '{"concepts": []}'}}]}, sink
    )
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    from ingestion.extractor import ConceptExtractor

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    extractor = ConceptExtractor(db)
    try:
        out = extractor._ollama_generate("prompt")
        assert out == '{"concepts": []}'
        assert sink["url"] == "https://api.openai.com/v1/chat/completions"
        headers = {k.lower(): v for k, v in sink["headers"].items()}
        assert headers["authorization"] == "Bearer sk-test-openai"
        assert sink["body"]["model"] == "gpt-4o-mini"
        assert sink["body"]["messages"] == [
            {"role": "user", "content": "prompt"}
        ]
    finally:
        db.close()


def test_extractor_defaults_to_ollama(monkeypatch):
    """Without provider env, ConceptExtractor hits the legacy Ollama URL."""
    import ingestion.extractor as extractor_mod

    sink: dict = {}

    def fake(req, timeout=None, context=None, **_kwargs):
        sink["url"] = req.full_url
        sink["timeout"] = timeout
        sink["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"response": "ok"})

    monkeypatch.setattr(extractor_mod.urllib.request, "urlopen", fake)

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    ex = extractor_mod.ConceptExtractor(db)
    try:
        assert ex._ollama_generate("p") == "ok"
        assert sink["url"] == "http://localhost:11434/api/generate"
        # Options carry num_predict — Ollama-native key
        assert "num_predict" in sink["body"]["options"]
    finally:
        db.close()


# ──────────────────────────────────────────────
# deep_enricher
# ──────────────────────────────────────────────


def test_deep_enricher_uses_configured_provider(monkeypatch):
    """MEMORY_ENRICH_PROVIDER=anthropic → Anthropic endpoint with x-api-key."""
    monkeypatch.setenv("MEMORY_ENRICH_PROVIDER", "anthropic")
    monkeypatch.setenv("MEMORY_LLM_API_KEY", "sk-ant-xyz")
    monkeypatch.setenv("MEMORY_ENRICH_MODEL", "claude-haiku-4-5")

    import llm_provider

    sink: dict = {}
    fake = _capture_urlopen(
        {"content": [{"type": "text", "text": "intent_fact"}]}, sink
    )
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    import deep_enricher

    out = deep_enricher._llm_complete("prompt", num_predict=30)
    assert out == "intent_fact"
    assert sink["url"] == "https://api.anthropic.com/v1/messages"
    headers = {k.lower(): v for k, v in sink["headers"].items()}
    assert headers["x-api-key"] == "sk-ant-xyz"
    assert headers["anthropic-version"] == "2023-06-01"
    assert sink["body"]["model"] == "claude-haiku-4-5"
    # Anthropic schema uses `messages`, not `prompt`.
    assert sink["body"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert sink["body"]["max_tokens"] == 30


def test_representations_uses_configured_provider(monkeypatch):
    """MEMORY_REPR_PROVIDER=openai with custom api_base → OpenRouter-style route."""
    monkeypatch.setenv("MEMORY_REPR_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_LLM_API_KEY", "sk-or-999")
    monkeypatch.setenv("MEMORY_LLM_API_BASE", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MEMORY_REPR_MODEL", "anthropic/claude-haiku-4.5")

    import llm_provider

    sink: dict = {}
    fake = _capture_urlopen(
        {"choices": [{"message": {"content": "short summary."}}]}, sink
    )
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    import representations

    out = representations._llm_complete("summarize me", num_predict=80)
    assert out == "short summary."
    assert sink["url"] == "https://openrouter.ai/api/v1/chat/completions"
    headers = {k.lower(): v for k, v in sink["headers"].items()}
    assert headers["authorization"] == "Bearer sk-or-999"
    assert sink["body"]["model"] == "anthropic/claude-haiku-4.5"
    assert sink["body"]["max_tokens"] == 80
    # temperature 0.2 is repr-specific (vs 0.1 for enrich/triple)
    assert sink["body"]["temperature"] == pytest.approx(0.2)


# ──────────────────────────────────────────────
# reflection.agent — graceful fallback on provider error
# ──────────────────────────────────────────────


def test_reflection_merge_fallback_to_ollama_on_openai_error(monkeypatch, tmp_path):
    """When the configured provider is unreachable, merge must degrade safely.

    With provider-aware has_llm()/available() the failure is now caught at
    the gate (has_llm() returns False → _make_llm_merge_fn returns None)
    instead of surfacing later as an empty merged string. Both outcomes make
    FactMerger skip the cluster cleanly.

    When we force the gate open, the closure must still swallow URLError at
    complete() time and return "" — which is what this test originally
    protected.
    """
    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_LLM_API_KEY", "sk-broken")

    import config
    import llm_provider

    # Force the has_llm() / provider.available() gate open so we can exercise
    # the runtime-error branch inside the returned closure.
    monkeypatch.setattr(config, "has_llm", lambda *a, **kw: True)
    monkeypatch.setattr(
        llm_provider.OpenAIProvider, "available", lambda self: True
    )

    def explode(req, timeout=None, context=None, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", explode)

    # Minimal DB good enough to build ReflectionAgent
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row

    from reflection.agent import ReflectionAgent

    agent = ReflectionAgent(db)
    try:
        merge = agent._make_llm_merge_fn()
        assert merge is not None, "gate forced open; closure must exist"
        # Call it — network blows up, but closure should swallow and return ""
        result = merge(["fact A about auth", "fact B about auth"])
        assert result == ""
    finally:
        db.close()
