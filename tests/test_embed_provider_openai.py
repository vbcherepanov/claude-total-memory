"""Day-1 LoCoMo wiring — OpenAI text-embedding-3-large path.

These tests are zero-network: every test that would hit `api.openai.com`
patches `urllib.request.urlopen` (the embed_provider transport) so the
call shape is asserted but never reaches the wire. The whole suite must
be runnable on a laptop with no `OPENAI_API_KEY` and no internet.
"""

from __future__ import annotations

import importlib
import io
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ──────────────────────────────────────────────
# Fake-HTTP plumbing (mirrors test_embed_provider.py shape)
# ──────────────────────────────────────────────


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def read(self):
        return self._payload


def _capture_urlopen(payload_factory, sink: dict):
    """Return a fake `urlopen` that records each call into `sink`.

    `payload_factory` is a callable `(call_idx, body) -> dict` so tests can
    return per-batch payloads sized to the actual request.
    """
    def fake(req, timeout=None, *, context=None, **_kw):
        body = json.loads(req.data.decode("utf-8")) if req.data else None
        idx = sink.setdefault("call_count", 0)
        sink["call_count"] = idx + 1
        sink.setdefault("urls", []).append(req.full_url)
        sink.setdefault("bodies", []).append(body)
        sink.setdefault("timeouts", []).append(timeout)
        payload = payload_factory(idx, body)
        return _FakeResp(payload)
    return fake


def _make_payload(n: int, dim: int = 8, base: float = 1.0) -> dict:
    """Generate a `data` payload with `n` non-zero vectors so L2-normalise has
    something meaningful to do (zero-vectors stay zero)."""
    items = []
    for i in range(n):
        # Distinct vectors so order-preservation / batching can be checked.
        vec = [base + float(i)] * dim
        items.append({"index": i, "embedding": vec})
    return {"data": items}


# ──────────────────────────────────────────────
# Test 1 — init reports correct dim per model
# ──────────────────────────────────────────────


def test_openai_embedder_init():
    """Constructing an OpenAIEmbedProvider for text-embedding-3-large
    reports 3072 dim *without* any network access."""
    import embed_provider

    large = embed_provider.OpenAIEmbedProvider(
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        model="text-embedding-3-large",
    )
    small = embed_provider.OpenAIEmbedProvider(
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        model="text-embedding-3-small",
    )
    ada = embed_provider.OpenAIEmbedProvider(
        api_key="sk-test",
        api_base="https://api.openai.com/v1",
        model="text-embedding-ada-002",
    )

    assert large.dim() == 3072
    assert small.dim() == 1536
    assert ada.dim() == 1536
    assert large.model == "text-embedding-3-large"
    assert large.available() is True


# ──────────────────────────────────────────────
# Test 2 — L2-normalisation produces unit-norm vectors
# ──────────────────────────────────────────────


def test_openai_embedder_l2_normalised(monkeypatch):
    """When the OpenAI API returns an unnormalised vector, the provider
    rescales it so ||v||_2 ≈ 1.0. Cosine similarity then collapses to a
    plain dot product, which the binary-quantization sign-bit search
    (and the float32 SQL path) both rely on."""
    import embed_provider

    sink: dict = {}
    raw_vec = [3.0, 4.0]  # ||v|| = 5.0, normalised → [0.6, 0.8]
    payload = {"data": [{"index": 0, "embedding": raw_vec}]}
    monkeypatch.setattr(
        embed_provider.urllib.request,
        "urlopen",
        _capture_urlopen(lambda *_: payload, sink),
    )

    p = embed_provider.OpenAIEmbedProvider(
        api_key="sk-x",
        api_base="https://api.openai.com/v1",
        model="text-embedding-3-large",
        normalize=True,
    )
    out = p.embed(["hello"])
    assert len(out) == 1
    norm = math.sqrt(sum(x * x for x in out[0]))
    assert abs(norm - 1.0) < 1e-9
    # Concrete values from the 3-4-5 triangle.
    assert abs(out[0][0] - 0.6) < 1e-9
    assert abs(out[0][1] - 0.8) < 1e-9


# ──────────────────────────────────────────────
# Test 3 — batching: 130 inputs at batch_size=64 → 3 API calls
# ──────────────────────────────────────────────


def test_openai_embedder_batches(monkeypatch):
    """OpenAI caps each /embeddings request at a fixed input count. The
    provider must transparently chunk a 130-item input into 3 calls (64,
    64, 2) and stitch the responses back together in input order."""
    import embed_provider

    sink: dict = {}

    def factory(idx, body):
        n = len(body["input"])
        return _make_payload(n)

    monkeypatch.setattr(
        embed_provider.urllib.request,
        "urlopen",
        _capture_urlopen(factory, sink),
    )

    p = embed_provider.OpenAIEmbedProvider(
        api_key="sk-x",
        api_base="https://api.openai.com/v1",
        model="text-embedding-3-large",
        batch_size=64,
        normalize=False,
    )
    inputs = [f"text-{i}" for i in range(130)]
    out = p.embed(inputs)

    assert len(out) == 130
    assert sink["call_count"] == 3
    assert [len(b["input"]) for b in sink["bodies"]] == [64, 64, 2]
    # Same model on every call — important when /v1/embeddings rejects mixed.
    assert {b["model"] for b in sink["bodies"]} == {"text-embedding-3-large"}


# ──────────────────────────────────────────────
# Test 4 — env-driven dispatch
# ──────────────────────────────────────────────


def test_dispatch_via_env_var(monkeypatch):
    """`MEMORY_EMBED_PROVIDER=openai` plus `MEMORY_EMBED_MODEL=…` must
    cause `provider_from_env()` to construct an OpenAIEmbedProvider with
    the right model and (production-default) L2-normalisation enabled."""
    import config as _cfg
    import embed_provider

    monkeypatch.setenv("MEMORY_EMBED_PROVIDER", "openai")
    monkeypatch.setenv("MEMORY_EMBED_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("MEMORY_EMBED_API_KEY", "sk-test-env")
    importlib.reload(_cfg)
    importlib.reload(embed_provider)

    prov = embed_provider.provider_from_env()
    assert isinstance(prov, embed_provider.OpenAIEmbedProvider)
    assert prov.model == "text-embedding-3-large"
    assert prov.dim() == 3072
    assert prov._normalize is True  # production hot-path default
    assert prov.api_key == "sk-test-env"


# ──────────────────────────────────────────────
# Test 5 — reembed --dry-run prints token estimate, no API call
# ──────────────────────────────────────────────


def test_reembed_dry_run(monkeypatch, tmp_path, capsys):
    """`scripts/reembed.py --dry-run --provider openai --model …` must
    report rows-to-process plus a token & USD cost estimate without
    touching the network. We also patch `urlopen` to raise on call so any
    accidental API hit fails the test loudly."""
    import sqlite3
    import urllib.request

    db_path = tmp_path / "memory.db"
    db = sqlite3.connect(str(db_path))
    db.executescript(
        """
        CREATE TABLE knowledge (id INTEGER PRIMARY KEY, content TEXT);
        CREATE TABLE embeddings (
            knowledge_id INTEGER PRIMARY KEY,
            binary_vector BLOB,
            float32_vector BLOB,
            embed_model TEXT,
            embed_dim INTEGER,
            created_at TEXT
        );
        """
    )
    # 3 rows — easy to reason about token estimate.
    db.execute("INSERT INTO knowledge VALUES (1, ?)", ("a" * 40,))   # ~10 tok
    db.execute("INSERT INTO knowledge VALUES (2, ?)", ("b" * 80,))   # ~20 tok
    db.execute("INSERT INTO knowledge VALUES (3, ?)", ("c" * 4,))    # ~1 tok
    db.commit()
    db.close()

    def boom(*a, **kw):
        raise AssertionError("dry-run must not perform any HTTP call")
    monkeypatch.setattr(urllib.request, "urlopen", boom)

    sys.path.insert(0, str(ROOT / "scripts"))
    if "reembed" in sys.modules:
        del sys.modules["reembed"]
    import reembed  # type: ignore

    rc = reembed.main([
        "--db", str(db_path),
        "--provider", "openai",
        "--model", "text-embedding-3-large",
        "--dry-run",
    ])
    out = capsys.readouterr().out

    assert rc == 2  # dry-run / no-confirm exit code per existing contract
    assert "model:    text-embedding-3-large" in out
    assert "rows to re-embed:    3" in out
    assert "estimated tokens:" in out
    assert "estimated cost:" in out
    assert "$0.13/1M tokens" in out


# ──────────────────────────────────────────────
# Test 6 — no real OpenAI client construction at import / wiring time
# ──────────────────────────────────────────────


def test_no_real_api_calls(monkeypatch):
    """Hard guarantee for CI: even if a developer mistakenly wires a real
    `openai.OpenAI()` somewhere, our provider construction path must not
    invoke it. We patch `openai.OpenAI` to raise on instantiation; the
    test must still pass — proving our path is HTTP-via-urllib (already
    mocked) and not the high-level SDK."""
    fake_openai = type(sys)("openai")

    def _raise(*_a, **_kw):
        raise AssertionError("real OpenAI() client should not be instantiated")
    fake_openai.OpenAI = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    import embed_provider
    importlib.reload(embed_provider)

    sink: dict = {}
    monkeypatch.setattr(
        embed_provider.urllib.request,
        "urlopen",
        _capture_urlopen(lambda *_: _make_payload(2, dim=4), sink),
    )

    p = embed_provider.OpenAIEmbedProvider(
        api_key="sk-x",
        api_base="https://api.openai.com/v1",
        model="text-embedding-3-large",
    )
    out = p.embed(["hello", "world"])
    assert len(out) == 2
    assert sink["call_count"] == 1
    # Importing/embedding never touched the patched OpenAI() — if it had,
    # the AssertionError above would have surfaced.


# ──────────────────────────────────────────────
# Test 7 — l2 helper edge cases (zero vector, already-unit vector)
# ──────────────────────────────────────────────


def test_l2_normalise_edge_cases():
    """Zero vector stays zero (no division blow-up); already-unit vector
    is left unchanged within float epsilon."""
    import embed_provider

    z = embed_provider._l2_normalise_vec([0.0, 0.0, 0.0])
    assert z == [0.0, 0.0, 0.0]

    unit = [1.0, 0.0, 0.0]
    out = embed_provider._l2_normalise_vec(unit)
    assert all(abs(a - b) < 1e-12 for a, b in zip(out, unit))


# ──────────────────────────────────────────────
# Test 8 — cost estimator math matches published rates
# ──────────────────────────────────────────────


def test_estimate_cost_matches_published_rate():
    """LoCoMo Day-1 sanity check — 1M tokens at $0.13/1M for the large
    model must yield exactly $0.13. The 16,500-row corpus mentioned in
    the roadmap (~5882 raw + 10487 synth + summaries) lands well under
    $1 once tokens are factored, but the unit math has to be correct
    first."""
    sys.path.insert(0, str(ROOT / "scripts"))
    if "reembed" in sys.modules:
        del sys.modules["reembed"]
    import reembed  # type: ignore

    assert abs(reembed.estimate_cost_usd("text-embedding-3-large", 1_000_000) - 0.13) < 1e-9
    assert abs(reembed.estimate_cost_usd("text-embedding-3-small", 1_000_000) - 0.02) < 1e-9
    assert reembed.estimate_cost_usd("unknown-model-xyz", 1_000_000) == 0.0

    # 4 chars/token heuristic: 400 chars across a list → ~100 tokens.
    assert reembed.estimate_tokens(["a" * 400]) == 100
    assert reembed.estimate_tokens([]) == 0
