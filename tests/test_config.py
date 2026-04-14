"""Tests for src/config.py — Ollama detection + LLM feature gating."""

from __future__ import annotations

import urllib.error

import pytest


# ──────────────────────────────────────────────
# detect_ollama — HTTP probe
# ──────────────────────────────────────────────


def test_detect_ollama_returns_true_when_responds(monkeypatch):
    from config import detect_ollama
    import config

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"models":[{"name":"qwen2.5-coder:7b"}]}'

    def fake_urlopen(req, timeout=2):
        return _R()

    monkeypatch.setattr(config.urllib.request, "urlopen", fake_urlopen)
    config._cache_clear()
    assert detect_ollama() is True


def test_detect_ollama_returns_false_when_unreachable(monkeypatch):
    from config import detect_ollama
    import config

    def boom(req, timeout=2):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(config.urllib.request, "urlopen", boom)
    config._cache_clear()
    assert detect_ollama() is False


def test_detect_ollama_returns_false_on_404(monkeypatch):
    from config import detect_ollama
    import config

    def boom(req, timeout=2):
        raise urllib.error.HTTPError("u", 404, "nf", {}, None)

    monkeypatch.setattr(config.urllib.request, "urlopen", boom)
    config._cache_clear()
    assert detect_ollama() is False


def test_detect_ollama_caches_result(monkeypatch):
    """Second call within TTL should not re-hit the network."""
    from config import detect_ollama
    import config

    calls = {"n": 0}

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"models":[]}'

    def fake_urlopen(req, timeout=2):
        calls["n"] += 1
        return _R()

    monkeypatch.setattr(config.urllib.request, "urlopen", fake_urlopen)
    config._cache_clear()

    detect_ollama()
    detect_ollama()
    detect_ollama()
    assert calls["n"] == 1


# ──────────────────────────────────────────────
# list_ollama_models / has_model
# ──────────────────────────────────────────────


def test_list_models_returns_names(monkeypatch):
    from config import list_ollama_models
    import config

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self):
            return b'{"models":[{"name":"qwen2.5-coder:7b"},{"name":"nomic-embed-text:latest"}]}'

    monkeypatch.setattr(config.urllib.request, "urlopen", lambda r, timeout=2: _R())
    config._cache_clear()
    names = list_ollama_models()
    assert "qwen2.5-coder:7b" in names
    assert "nomic-embed-text:latest" in names


def test_has_model_matches_exact_or_prefix(monkeypatch):
    from config import has_model
    import config

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self):
            return b'{"models":[{"name":"qwen2.5-coder:7b"},{"name":"vitalii-brain:latest"}]}'

    monkeypatch.setattr(config.urllib.request, "urlopen", lambda r, timeout=2: _R())
    config._cache_clear()

    assert has_model("qwen2.5-coder:7b") is True
    assert has_model("qwen2.5-coder") is True       # prefix without tag
    assert has_model("vitalii-brain") is True
    assert has_model("gpt-4") is False


def test_has_model_returns_false_when_ollama_down(monkeypatch):
    from config import has_model
    import config

    def boom(r, timeout=2):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(config.urllib.request, "urlopen", boom)
    config._cache_clear()
    assert has_model("anything") is False


# ──────────────────────────────────────────────
# has_llm() — top-level gate
# ──────────────────────────────────────────────


def test_has_llm_true_when_ollama_and_model_present(monkeypatch):
    import config

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"models":[{"name":"qwen2.5-coder:7b"}]}'

    monkeypatch.setattr(config.urllib.request, "urlopen", lambda r, timeout=2: _R())
    monkeypatch.setenv("MEMORY_LLM_MODEL", "qwen2.5-coder:7b")
    config._cache_clear()
    assert config.has_llm() is True


def test_has_llm_false_when_ollama_down(monkeypatch):
    import config

    def boom(r, timeout=2):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(config.urllib.request, "urlopen", boom)
    monkeypatch.setenv("MEMORY_LLM_MODEL", "qwen2.5-coder:7b")
    config._cache_clear()
    assert config.has_llm() is False


def test_has_llm_false_when_model_missing(monkeypatch):
    import config

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"models":[{"name":"nomic-embed-text:latest"}]}'

    monkeypatch.setattr(config.urllib.request, "urlopen", lambda r, timeout=2: _R())
    monkeypatch.setenv("MEMORY_LLM_MODEL", "qwen2.5-coder:7b")
    config._cache_clear()
    assert config.has_llm() is False


def test_has_llm_can_be_force_disabled(monkeypatch):
    import config
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "false")
    config._cache_clear()
    assert config.has_llm() is False


def test_has_llm_can_be_force_enabled_for_tests(monkeypatch):
    """Force-enabled mode skips Ollama probe (used in unit tests with stubs)."""
    import config
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "force")
    config._cache_clear()
    assert config.has_llm() is True


# ──────────────────────────────────────────────
# get_llm_model — falls back gracefully
# ──────────────────────────────────────────────


def test_get_llm_model_returns_env_default():
    from config import get_llm_model
    assert get_llm_model() != ""


def test_get_llm_model_respects_env(monkeypatch):
    from config import get_llm_model
    monkeypatch.setenv("MEMORY_LLM_MODEL", "vitalii-brain")
    assert get_llm_model() == "vitalii-brain"


# ──────────────────────────────────────────────
# get_status — for /api/status / dashboard
# ──────────────────────────────────────────────


def test_get_status_reports_full_picture(monkeypatch):
    import config

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b'{"models":[{"name":"qwen2.5-coder:7b"}]}'

    monkeypatch.setattr(config.urllib.request, "urlopen", lambda r, timeout=2: _R())
    monkeypatch.setenv("MEMORY_LLM_MODEL", "qwen2.5-coder:7b")
    config._cache_clear()

    s = config.get_status()
    assert s["ollama_available"] is True
    assert s["model_configured"] == "qwen2.5-coder:7b"
    assert s["model_installed"] is True
    assert s["llm_enabled"] is True
    assert "qwen2.5-coder:7b" in s["installed_models"]
