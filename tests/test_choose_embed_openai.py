"""v9.0 D1 — OpenAI embedding backend wiring through choose_embed."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for k in (
        "V9_EMBED_BACKEND",
        "V9_LOCOMO_TUNED_PATH",
        "MEMORY_EMBED_API_KEY",
        "MEMORY_EMBED_API_BASE",
        "MEMORY_EMBED_PROVIDER",
        "MEMORY_EMBED_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def _fresh():
    import config as _cfg
    import choose_embed as _ch
    importlib.reload(_cfg)
    importlib.reload(_ch)
    return _cfg, _ch


def test_openai_3_large_resolves_correctly():
    cfg, ch = _fresh()
    cfg_module = importlib.import_module("config")
    cfg_module.os.environ["V9_EMBED_BACKEND"] = "openai-3-large"
    importlib.reload(cfg)
    importlib.reload(ch)

    assert ch.resolve_backend() == "openai-3-large"
    assert ch.resolve_model_name() == "text-embedding-3-large"
    assert ch.backend_dim() == 3072


def test_openai_3_small_resolves_correctly(monkeypatch):
    monkeypatch.setenv("V9_EMBED_BACKEND", "openai-3-small")
    cfg, ch = _fresh()
    assert ch.resolve_backend() == "openai-3-small"
    assert ch.resolve_model_name() == "text-embedding-3-small"
    assert ch.backend_dim() == 1536


def test_invalid_backend_falls_back_to_fastembed(monkeypatch):
    monkeypatch.setenv("V9_EMBED_BACKEND", "openai-totally-fake")
    cfg, ch = _fresh()
    assert ch.resolve_backend() == "fastembed"


def test_get_provider_returns_openai_provider_class(monkeypatch):
    monkeypatch.setenv("V9_EMBED_BACKEND", "openai-3-large")
    cfg, ch = _fresh()
    prov = ch.get_provider()
    assert type(prov).__name__ == "OpenAIEmbedProvider"
    assert prov.dim() == 3072


def test_get_provider_unavailable_without_key(monkeypatch):
    monkeypatch.setenv("V9_EMBED_BACKEND", "openai-3-large")
    cfg, ch = _fresh()
    prov = ch.get_provider()
    assert prov.available() is False


def test_get_provider_available_with_key(monkeypatch):
    monkeypatch.setenv("V9_EMBED_BACKEND", "openai-3-large")
    monkeypatch.setenv("MEMORY_EMBED_API_KEY", "sk-test")
    cfg, ch = _fresh()
    prov = ch.get_provider()
    assert prov.available() is True


def test_locomo_tuned_path_override(monkeypatch):
    monkeypatch.setenv("V9_EMBED_BACKEND", "locomo-tuned-minilm")
    monkeypatch.setenv("V9_LOCOMO_TUNED_PATH", "/abs/path/model")
    cfg, ch = _fresh()
    assert ch.resolve_model_name() == "/abs/path/model"


def test_supported_backends_listed_explicitly():
    cfg, _ = _fresh()
    backends = set(cfg._SUPPORTED_V9_EMBED_BACKENDS)
    assert {"openai-3-small", "openai-3-large", "locomo-tuned-minilm"} <= backends
    assert {"fastembed", "minilm", "bge-m3", "e5-large"} <= backends
