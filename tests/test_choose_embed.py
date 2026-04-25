"""Tests for src/choose_embed.py — v9.0 B1 embedding backend selector."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────
# Fake fastembed TextEmbedding — used to prevent real model downloads
# ──────────────────────────────────────────────


class _FakeVec:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return list(self._vals)


class _FakeTextEmbedding:
    """Captures model_name and yields predictable vectors."""

    last_model_name: str | None = None

    def __init__(self, model_name, *args, **kwargs):
        _FakeTextEmbedding.last_model_name = model_name
        self.model_name = model_name

    def embed(self, texts):
        for i, t in enumerate(texts):
            yield _FakeVec([float(i), float(len(t)), 0.25])


@pytest.fixture
def fake_fastembed(monkeypatch):
    """Install a fake fastembed module so no model gets downloaded."""
    fake_module = type(sys)("fastembed")
    fake_module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fake_module)
    _FakeTextEmbedding.last_model_name = None
    yield _FakeTextEmbedding


@pytest.fixture(autouse=True)
def _reset_v9_env(monkeypatch):
    """Keep tests hermetic re: V9_EMBED_BACKEND."""
    monkeypatch.delenv("V9_EMBED_BACKEND", raising=False)
    yield


# ──────────────────────────────────────────────
# 1. Default backend → fastembed MiniLM
# ──────────────────────────────────────────────


def test_default_backend_is_fastembed_minilm(fake_fastembed):
    import choose_embed

    assert choose_embed.resolve_backend() == "fastembed"
    assert (
        choose_embed.resolve_model_name()
        == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    p = choose_embed.get_provider()
    # Trigger model init via embed() to confirm correct model name is used.
    out = p.embed(["abc"])
    assert out == [[0.0, 3.0, 0.25]]
    assert (
        fake_fastembed.last_model_name
        == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ──────────────────────────────────────────────
# 2. V9_EMBED_BACKEND=bge-m3 → sentence-transformers with BAAI/bge-m3
# ──────────────────────────────────────────────


def test_bge_m3_backend_uses_sentence_transformers(monkeypatch):
    import choose_embed

    monkeypatch.setenv("V9_EMBED_BACKEND", "bge-m3")

    captured: dict = {}

    class _FakeST:
        def __init__(self, model_name, *a, **kw):
            captured["model_name"] = model_name

        def encode(self, texts, **kw):
            # Return a list of list-like objects.
            return [[float(i), 0.1, 0.2, 0.3] for i, _ in enumerate(texts)]

    fake_st = type(sys)("sentence_transformers")
    fake_st.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    assert choose_embed.resolve_backend() == "bge-m3"
    assert choose_embed.resolve_model_name() == "BAAI/bge-m3"
    p = choose_embed.get_provider()
    assert isinstance(p, choose_embed.SentenceTransformersProvider)
    out = p.embed(["foo", "bar"])
    assert out == [[0.0, 0.1, 0.2, 0.3], [1.0, 0.1, 0.2, 0.3]]
    assert captured["model_name"] == "BAAI/bge-m3"
    assert p.dim() == 1024  # expected dim preloaded


# ──────────────────────────────────────────────
# 3. V9_EMBED_BACKEND=e5-large → fastembed with intfloat/multilingual-e5-large
# ──────────────────────────────────────────────


def test_e5_large_backend(monkeypatch, fake_fastembed):
    import choose_embed

    monkeypatch.setenv("V9_EMBED_BACKEND", "e5-large")

    assert choose_embed.resolve_backend() == "e5-large"
    assert choose_embed.resolve_model_name() == "intfloat/multilingual-e5-large"
    p = choose_embed.get_provider()
    p.embed(["x"])
    assert fake_fastembed.last_model_name == "intfloat/multilingual-e5-large"


# ──────────────────────────────────────────────
# 4. V9_EMBED_BACKEND=minilm → explicit alias for fastembed
# ──────────────────────────────────────────────


def test_minilm_backend_is_explicit_alias(monkeypatch, fake_fastembed):
    import choose_embed

    monkeypatch.setenv("V9_EMBED_BACKEND", "minilm")

    assert choose_embed.resolve_backend() == "minilm"
    assert (
        choose_embed.resolve_model_name()
        == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    p = choose_embed.get_provider()
    p.embed(["hi"])
    assert (
        fake_fastembed.last_model_name
        == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ──────────────────────────────────────────────
# 5. Invalid value → falls back to fastembed default
# ──────────────────────────────────────────────


def test_invalid_backend_falls_back_to_fastembed(monkeypatch, fake_fastembed):
    import choose_embed

    monkeypatch.setenv("V9_EMBED_BACKEND", "not-a-real-backend")

    assert choose_embed.resolve_backend() == "fastembed"
    p = choose_embed.get_provider()
    p.embed(["q"])
    assert (
        fake_fastembed.last_model_name
        == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ──────────────────────────────────────────────
# 6. Module import succeeds when ST / fastembed libs unavailable
# ──────────────────────────────────────────────


def test_module_import_survives_missing_backends(monkeypatch):
    # Shadow both libs as import-failing.
    monkeypatch.setitem(sys.modules, "fastembed", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    # Force a fresh reimport so _ep/choose_embed don't cache the real libs.
    sys.modules.pop("choose_embed", None)
    sys.modules.pop("embed_provider", None)

    import choose_embed  # noqa: F401

    # BGE-M3 provider must be constructible without raising at import/build.
    p = choose_embed.SentenceTransformersProvider(model="BAAI/bge-m3", expected_dim=1024)
    assert p.available() is False
    with pytest.raises(RuntimeError, match="unavailable"):
        p.embed(["x"])

    # FastEmbed provider must also stay build-able + report unavailable.
    monkeypatch.setenv("V9_EMBED_BACKEND", "fastembed")
    q = choose_embed.get_provider()
    assert q.available() is False


# ──────────────────────────────────────────────
# 7. override argument beats env var (used by benchmarks)
# ──────────────────────────────────────────────


def test_override_argument_overrides_env(monkeypatch, fake_fastembed):
    import choose_embed

    monkeypatch.setenv("V9_EMBED_BACKEND", "e5-large")
    # Explicit override wins.
    assert choose_embed.resolve_backend("minilm") == "minilm"
    assert choose_embed.resolve_model_name("minilm") == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


# ──────────────────────────────────────────────
# 8. backend_dim returns expected sizes
# ──────────────────────────────────────────────


def test_backend_dim_lookup():
    import choose_embed

    assert choose_embed.backend_dim("fastembed") == 384
    assert choose_embed.backend_dim("minilm") == 384
    assert choose_embed.backend_dim("e5-large") == 1024
    assert choose_embed.backend_dim("bge-m3") == 1024
    # Invalid → fallback to fastembed.
    assert choose_embed.backend_dim("nonsense") == 384
