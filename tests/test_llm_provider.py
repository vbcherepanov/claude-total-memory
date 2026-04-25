"""Tests for src/llm_provider.py — cloud LLM provider abstraction.

All HTTP traffic is mocked via urllib.request.urlopen. No real network.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure src/ on path (conftest already does this but keep explicit)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


def _capture_urlopen(payload: dict, sink: dict):
    """Return a fake urlopen that records the request it received."""
    def fake(req, timeout=None, context=None, **_kwargs):
        sink["url"] = req.full_url
        sink["headers"] = dict(req.headers)
        sink["body"] = json.loads(req.data.decode("utf-8")) if req.data else None
        sink["timeout"] = timeout
        sink["ssl_context"] = context
        return _FakeResp(payload)
    return fake


# ──────────────────────────────────────────────
# OllamaProvider
# ──────────────────────────────────────────────


def test_ollama_provider_roundtrip(monkeypatch):
    import llm_provider

    sink: dict = {}
    fake = _capture_urlopen({"response": "  hello world  "}, sink)
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    provider = llm_provider.OllamaProvider(
        api_base="http://localhost:11434", model="qwen2.5-coder:7b"
    )
    out = provider.complete("hi", max_tokens=42, temperature=0.5, timeout=7.0)

    assert out == "hello world"
    assert sink["url"] == "http://localhost:11434/api/generate"
    assert sink["body"]["model"] == "qwen2.5-coder:7b"
    assert sink["body"]["prompt"] == "hi"
    assert sink["body"]["stream"] is False
    assert sink["body"]["options"]["num_predict"] == 42
    assert sink["body"]["options"]["temperature"] == 0.5
    assert sink["timeout"] == 7.0
    # Ollama does not require auth
    assert "Authorization" not in sink["headers"]


def test_ollama_provider_model_override(monkeypatch):
    import llm_provider

    sink: dict = {}
    fake = _capture_urlopen({"response": "ok"}, sink)
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    provider = llm_provider.OllamaProvider(api_base="http://x:1", model="default-model")
    provider.complete("p", model="override")
    assert sink["body"]["model"] == "override"


# ──────────────────────────────────────────────
# OpenAIProvider
# ──────────────────────────────────────────────


def test_openai_provider_roundtrip(monkeypatch):
    import llm_provider

    sink: dict = {}
    payload = {
        "choices": [
            {"message": {"role": "assistant", "content": " hello from gpt "}}
        ]
    }
    fake = _capture_urlopen(payload, sink)
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    provider = llm_provider.OpenAIProvider(
        api_key="sk-test", api_base="https://api.openai.com/v1", model="gpt-4o-mini"
    )
    out = provider.complete("question", max_tokens=100, temperature=0.2, timeout=5.0)

    assert out == "hello from gpt"
    assert sink["url"] == "https://api.openai.com/v1/chat/completions"
    # urllib.Request lowercases header names in .headers mapping
    auth = {k.lower(): v for k, v in sink["headers"].items()}
    assert auth["authorization"] == "Bearer sk-test"
    assert sink["body"]["model"] == "gpt-4o-mini"
    assert sink["body"]["messages"] == [{"role": "user", "content": "question"}]
    assert sink["body"]["max_tokens"] == 100
    assert sink["body"]["temperature"] == 0.2


def test_openai_provider_custom_base_openrouter(monkeypatch):
    """OpenRouter works out of the box with the same provider."""
    import llm_provider

    sink: dict = {}
    payload = {"choices": [{"message": {"content": "routed"}}]}
    fake = _capture_urlopen(payload, sink)
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    provider = llm_provider.OpenAIProvider(
        api_key="sk-or-123",
        api_base="https://openrouter.ai/api/v1",
        model="anthropic/claude-haiku-4.5",
    )
    assert provider.available() is True
    assert provider.complete("hello") == "routed"
    assert sink["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert sink["body"]["model"] == "anthropic/claude-haiku-4.5"


def test_openai_provider_unavailable_without_key():
    import llm_provider

    provider = llm_provider.OpenAIProvider(
        api_key=None, api_base="https://api.openai.com/v1"
    )
    assert provider.available() is False
    with pytest.raises(RuntimeError, match="missing api_key"):
        provider.complete("x")


def test_openai_provider_malformed_response(monkeypatch):
    import llm_provider

    sink: dict = {}
    fake = _capture_urlopen({"choices": []}, sink)
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    provider = llm_provider.OpenAIProvider(
        api_key="sk", api_base="https://x/v1", model="m"
    )
    with pytest.raises(RuntimeError, match="malformed"):
        provider.complete("x")


# ──────────────────────────────────────────────
# AnthropicProvider
# ──────────────────────────────────────────────


def test_anthropic_provider_roundtrip(monkeypatch):
    import llm_provider

    sink: dict = {}
    payload = {"content": [{"type": "text", "text": " hi from claude "}]}
    fake = _capture_urlopen(payload, sink)
    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", fake)

    provider = llm_provider.AnthropicProvider(
        api_key="sk-ant-abc",
        api_base="https://api.anthropic.com/v1",
        model="claude-haiku-4-5",
    )
    out = provider.complete("hello", max_tokens=256, temperature=0.1, timeout=9.0)

    assert out == "hi from claude"
    assert sink["url"] == "https://api.anthropic.com/v1/messages"
    h = {k.lower(): v for k, v in sink["headers"].items()}
    assert h["x-api-key"] == "sk-ant-abc"
    assert h["anthropic-version"] == "2023-06-01"
    # Never Authorization header for Anthropic
    assert "authorization" not in h
    assert sink["body"]["model"] == "claude-haiku-4-5"
    assert sink["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert sink["body"]["max_tokens"] == 256
    assert sink["timeout"] == 9.0


def test_anthropic_provider_skips_non_text_blocks(monkeypatch):
    import llm_provider

    sink: dict = {}
    payload = {
        "content": [
            {"type": "tool_use", "name": "x"},
            {"type": "text", "text": "picked"},
        ]
    }
    monkeypatch.setattr(
        llm_provider.urllib.request, "urlopen", _capture_urlopen(payload, sink)
    )

    provider = llm_provider.AnthropicProvider(
        api_key="sk", api_base="https://a/v1", model="m"
    )
    assert provider.complete("x") == "picked"


def test_anthropic_provider_unavailable_without_key():
    import llm_provider

    provider = llm_provider.AnthropicProvider(
        api_key=None, api_base="https://api.anthropic.com/v1"
    )
    assert provider.available() is False
    with pytest.raises(RuntimeError, match="missing api_key"):
        provider.complete("x")


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


def test_make_provider_factory_ollama(monkeypatch):
    import llm_provider

    monkeypatch.delenv("MEMORY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    p = llm_provider.make_provider("ollama")
    assert isinstance(p, llm_provider.OllamaProvider)
    assert p.name == "ollama"


def test_make_provider_factory_openai_with_kwargs():
    import llm_provider

    p = llm_provider.make_provider(
        "openai",
        api_key="sk-x",
        api_base="https://openrouter.ai/api/v1",
        model="anthropic/claude-haiku-4.5",
    )
    assert isinstance(p, llm_provider.OpenAIProvider)
    assert p.api_key == "sk-x"
    assert p.api_base == "https://openrouter.ai/api/v1"
    assert p._default_model == "anthropic/claude-haiku-4.5"
    assert p.name == "openai"


def test_make_provider_factory_anthropic_reads_env(monkeypatch):
    import llm_provider
    import config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    monkeypatch.delenv("MEMORY_LLM_API_KEY", raising=False)

    p = llm_provider.make_provider("anthropic")
    assert isinstance(p, llm_provider.AnthropicProvider)
    assert p.api_key == "sk-ant-env"
    assert p.api_base == "https://api.anthropic.com/v1"


def test_make_provider_factory_auto_resolves(monkeypatch):
    import llm_provider

    monkeypatch.setenv("MEMORY_LLM_PROVIDER", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_LLM_API_KEY", raising=False)

    p = llm_provider.make_provider("auto")
    assert isinstance(p, llm_provider.OpenAIProvider)


def test_make_provider_factory_unknown():
    import llm_provider

    with pytest.raises(ValueError, match="unknown LLM provider"):
        llm_provider.make_provider("bogus")


def test_provider_unavailable_when_no_api_key(monkeypatch):
    import llm_provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_LLM_API_KEY", raising=False)

    p = llm_provider.make_provider("openai")
    assert isinstance(p, llm_provider.OpenAIProvider)
    assert p.available() is False
