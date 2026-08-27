"""Integration tests for EmbeddingProvider wiring into Store + reranker.

Covers:
  - default (fastembed) path still works without env — local call, dim=384
  - MEMORY_EMBED_PROVIDER=openai routes Store._embed_text through HTTPS
    POST to api.openai.com/v1/embeddings with captured headers/body
  - reranker._provider_embed picks up configured provider (OpenAI)
  - reranker with MEMORY_EMBED_PROVIDER=cohere hits api.cohere.com/v2/embed
  - dim mismatch between stored embeddings and provider raises RuntimeError
    with a clear re-embed hint
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


SRC_DIR = str(Path(__file__).parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ──────────────────────────────────────────────
# Helpers — fake urlopen (same shape as tests/test_embed_provider.py)
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
    # See note in test_embed_provider.py — production embed_provider
    # passes context=ssl_context for certifi compatibility, so the
    # mock has to accept it (and any forward-compatible kwargs).
    def fake(req, timeout=None, *, context=None, **_kw):
        sink["url"] = req.full_url
        sink["headers"] = dict(req.headers)
        sink["body"] = json.loads(req.data.decode("utf-8")) if req.data else None
        sink["timeout"] = timeout
        sink["context"] = context
        return _FakeResp(payload)
    return fake


def _fresh_store(monkeypatch, tmp_path):
    """Instantiate Store on a dedicated temp MEMORY_DIR.

    Importantly: we import `server` anew after each env tweak by clearing
    the module cache so Store.__init__ re-reads MEMORY_EMBED_PROVIDER.
    """
    import importlib

    # Drop cached server/embed modules so fresh env is picked up.
    for mod in ("server", "embed_provider", "reranker", "config"):
        sys.modules.pop(mod, None)

    import server  # noqa: E402
    importlib.reload(server)
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    return server.Store()


# ──────────────────────────────────────────────
# Default backward-compat path (FastEmbed)
# ──────────────────────────────────────────────


def test_embed_wiring_defaults_to_fastembed(monkeypatch, tmp_path):
    """With no env set, Store.embed() must go through the legacy fastembed
    path and return real 384-dim vectors."""
    monkeypatch.delenv("MEMORY_EMBED_PROVIDER", raising=False)
    monkeypatch.delenv("MEMORY_EMBED_MODEL", raising=False)
    monkeypatch.delenv("MEMORY_EMBED_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_EMBED_API_BASE", raising=False)

    store = _fresh_store(monkeypatch, tmp_path)
    try:
        # Label should still read "fastembed" so legacy branches keep working.
        assert store._embed_mode == "fastembed"

        vecs = store.embed(["hello world"])
        assert vecs is not None
        assert len(vecs) == 1
        # MiniLM-L12-v2 produces 384-dim vectors.
        assert len(vecs[0]) == 384
        assert all(isinstance(x, float) for x in vecs[0])
    finally:
        try:
            store.db.close()
        except Exception:
            pass


def _unit(vec: list[float]) -> list[float]:
    """L2-normalise, mirroring embed_provider._l2_normalise_vec."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


# ──────────────────────────────────────────────
# OpenAI wiring through Store
# ──────────────────────────────────────────────


def test_embed_wiring_uses_configured_provider(monkeypatch, tmp_path):
    """MEMORY_EMBED_PROVIDER=openai must reroute Store.embed() to the
    OpenAI /embeddings endpoint (no fastembed call)."""
    monkeypatch.setenv("MEMORY_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_EMBED_API_KEY", "sk-XXX")
    monkeypatch.setenv("MEMORY_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMORY_EMBED_API_BASE", "https://api.openai.com/v1")

    store = _fresh_store(monkeypatch, tmp_path)
    try:
        assert store._embed_mode == "openai"

        sink: dict = {}
        payload = {"data": [{"index": 0, "embedding": [0.11, 0.22, 0.33]}]}
        import embed_provider  # loaded by Store already
        monkeypatch.setattr(
            embed_provider.urllib.request, "urlopen",
            _capture_urlopen(payload, sink),
        )

        vecs = store.embed(["hello"])
        # The production OpenAI path L2-normalises so cosine == dot product
        # (see embed_provider._l2_normalise_vec); assert the unit vector the
        # stub payload maps to, not the raw API values.
        assert vecs == [_unit([0.11, 0.22, 0.33])]
        assert sink["url"] == "https://api.openai.com/v1/embeddings"
        h = {k.lower(): v for k, v in sink["headers"].items()}
        assert h["authorization"] == "Bearer sk-XXX"
        assert sink["body"] == {"input": ["hello"], "model": "text-embedding-3-small"}
    finally:
        try:
            store.db.close()
        except Exception:
            pass


# ──────────────────────────────────────────────
# Reranker wiring
# ──────────────────────────────────────────────


def test_reranker_uses_embed_provider(monkeypatch):
    """reranker._provider_embed must dispatch to the configured provider
    (OpenAI here) — no call to the legacy Ollama /api/embeddings endpoint."""
    monkeypatch.setenv("MEMORY_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_EMBED_API_KEY", "sk-test")
    monkeypatch.setenv("MEMORY_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMORY_EMBED_API_BASE", "https://api.openai.com/v1")

    for mod in ("embed_provider", "reranker", "config"):
        sys.modules.pop(mod, None)

    import reranker  # noqa: E402
    reranker._reset_embed_provider()

    sink: dict = {}
    payload = {"data": [{"index": 0, "embedding": [1.5, 2.5]}]}
    import embed_provider
    monkeypatch.setattr(
        embed_provider.urllib.request, "urlopen",
        _capture_urlopen(payload, sink),
    )

    vec = reranker._provider_embed("hypothetical answer")
    assert vec == _unit([1.5, 2.5])
    assert sink["url"] == "https://api.openai.com/v1/embeddings"
    assert sink["body"]["model"] == "text-embedding-3-small"


def test_reranker_cohere_provider(monkeypatch):
    """MEMORY_EMBED_PROVIDER=cohere must hit the v2 /embed endpoint."""
    monkeypatch.setenv("MEMORY_EMBED_PROVIDER", "cohere")
    monkeypatch.setenv("MEMORY_EMBED_API_KEY", "co-test")
    monkeypatch.setenv("MEMORY_EMBED_MODEL", "embed-multilingual-v3.0")
    monkeypatch.setenv("MEMORY_EMBED_API_BASE", "https://api.cohere.com/v2")

    for mod in ("embed_provider", "reranker", "config"):
        sys.modules.pop(mod, None)

    import reranker  # noqa: E402
    reranker._reset_embed_provider()

    sink: dict = {}
    payload = {"embeddings": {"float": [[0.1, 0.2, 0.3]]}}
    import embed_provider
    monkeypatch.setattr(
        embed_provider.urllib.request, "urlopen",
        _capture_urlopen(payload, sink),
    )

    vec = reranker._provider_embed("some text")
    assert vec == [0.1, 0.2, 0.3]
    assert sink["url"] == "https://api.cohere.com/v2/embed"
    assert sink["body"]["model"] == "embed-multilingual-v3.0"
    h = {k.lower(): v for k, v in sink["headers"].items()}
    assert h["authorization"] == "Bearer co-test"


# ──────────────────────────────────────────────
# Safety gate: dim mismatch
# ──────────────────────────────────────────────


def test_dim_mismatch_raises_clear_error(monkeypatch, tmp_path):
    """If the embeddings table already has rows with a different dim than
    the configured provider's dim(), Store.__init__ must refuse to boot
    and point the operator at tools/reembed.py."""
    # Step 1: bring up Store under fastembed (default) to seed the table
    # with one 384-dim embedding.
    monkeypatch.delenv("MEMORY_EMBED_PROVIDER", raising=False)
    monkeypatch.delenv("MEMORY_EMBED_MODEL", raising=False)
    monkeypatch.delenv("MEMORY_EMBED_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_EMBED_API_BASE", raising=False)

    store = _fresh_store(monkeypatch, tmp_path)
    try:
        # Fake an embedding row with dim=384 (FastEmbed MiniLM).
        import struct
        dim = 384
        vec = [0.01] * dim
        blob = struct.pack(f"{dim}f", *vec)
        # binary vector = packbits of sign bits, N/8 bytes
        import numpy as np
        bits = (np.array(vec, dtype=np.float32) > 0).astype(np.uint8)
        bin_blob = np.packbits(bits).tobytes()
        store.db.execute(
            "INSERT INTO embeddings (knowledge_id, binary_vector, float32_vector, "
            "embed_model, embed_dim, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (1, bin_blob, blob, "fastembed-test", dim, "2026-04-19T00:00:00Z"),
        )
        store.db.commit()
    finally:
        try:
            store.db.close()
        except Exception:
            pass

    # Step 2: switch to OpenAI (1536d) on the SAME MEMORY_DIR → boot must fail.
    monkeypatch.setenv("MEMORY_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_EMBED_API_KEY", "sk-x")
    monkeypatch.setenv("MEMORY_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("MEMORY_EMBED_API_BASE", "https://api.openai.com/v1")

    for mod in ("server", "embed_provider", "config"):
        sys.modules.pop(mod, None)

    import server  # noqa: E402
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        server.Store()
    msg = str(excinfo.value)
    assert "Embedding dimension mismatch" in msg
    assert "stored=384" in msg
    assert "provider=1536" in msg
    assert "reembed.py" in msg
