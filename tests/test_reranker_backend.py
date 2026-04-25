"""v9.0 D4 — pluggable reranker backend resolution.

Tests cover the dispatch layer only — actual model loading is mocked, so
the suite runs without sentence-transformers / FlagEmbedding installed.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for k in (
        "V9_RERANKER_BACKEND",
        "V9_RERANKER_MODEL",
        "V9_RERANKER_FP16",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def _fresh_modules():
    """Reload config + reranker so env changes take effect."""
    import config as _cfg
    import reranker as _rr
    importlib.reload(_cfg)
    importlib.reload(_rr)
    _rr._reset_reranker_cache()
    return _cfg, _rr


def test_default_backend_is_ce_marco():
    cfg, rr = _fresh_modules()
    assert cfg.get_v9_reranker_backend() == "ce-marco"
    assert rr._resolve_reranker_backend() == "ce-marco"
    assert rr._resolve_reranker_model("ce-marco") == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_bge_v2_m3_backend(monkeypatch):
    monkeypatch.setenv("V9_RERANKER_BACKEND", "bge-v2-m3")
    cfg, rr = _fresh_modules()
    assert cfg.get_v9_reranker_backend() == "bge-v2-m3"
    assert rr._resolve_reranker_model("bge-v2-m3") == "BAAI/bge-reranker-v2-m3"


def test_bge_large_backend(monkeypatch):
    monkeypatch.setenv("V9_RERANKER_BACKEND", "bge-large")
    cfg, rr = _fresh_modules()
    assert rr._resolve_reranker_model("bge-large") == "BAAI/bge-reranker-large"


def test_invalid_backend_falls_back(monkeypatch):
    monkeypatch.setenv("V9_RERANKER_BACKEND", "totally-fake")
    cfg, _ = _fresh_modules()
    assert cfg.get_v9_reranker_backend() == "ce-marco"


def test_off_backend_skips_reranking(monkeypatch):
    monkeypatch.setenv("V9_RERANKER_BACKEND", "off")
    _, rr = _fresh_modules()
    results = [
        {"r": {"content": "alpha", "project": "p"}, "score": 0.5},
        {"r": {"content": "beta", "project": "p"}, "score": 0.3},
    ]
    out = rr.rerank_results("query", results, top_k=2)
    assert [x["score"] for x in out] == [0.5, 0.3]


def test_model_override_wins(monkeypatch):
    monkeypatch.setenv("V9_RERANKER_BACKEND", "bge-v2-m3")
    monkeypatch.setenv("V9_RERANKER_MODEL", "custom/some-model")
    _, rr = _fresh_modules()
    assert rr._resolve_reranker_model("bge-v2-m3") == "custom/some-model"


def test_fp16_default_true(monkeypatch):
    cfg, _ = _fresh_modules()
    assert cfg.get_v9_reranker_use_fp16() is True


def test_fp16_disable(monkeypatch):
    monkeypatch.setenv("V9_RERANKER_FP16", "0")
    cfg, _ = _fresh_modules()
    assert cfg.get_v9_reranker_use_fp16() is False


def test_empty_results_short_circuit(monkeypatch):
    _, rr = _fresh_modules()
    assert rr.rerank_results("q", [], top_k=5) == []


def test_dispatch_uses_flag_kind_for_bge(monkeypatch):
    """If the FlagReranker stub exposes compute_score, kind=='flag'."""
    monkeypatch.setenv("V9_RERANKER_BACKEND", "bge-v2-m3")
    _, rr = _fresh_modules()

    calls = {"count": 0}

    class _StubFlag:
        def compute_score(self, pairs):  # noqa: D401
            calls["count"] += 1
            return [float(len(p[1])) for p in pairs]

    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: _StubFlag())
    rr._reset_reranker_cache()

    model, kind = rr._get_reranker("bge-v2-m3")
    assert kind == "flag"
    assert hasattr(model, "compute_score")

    # Use diverging original scores so the boost-only blend can move things.
    # Doc B has stronger CE signal AND higher orig score → must stay/become #1.
    results = [
        {"r": {"content": "short", "project": "p"}, "score": 0.4},
        {"r": {"content": "much-longer-content-here", "project": "p"}, "score": 0.5},
    ]
    out = rr.rerank_results("q", results, top_k=2)
    assert calls["count"] == 1
    assert out[0]["r"]["content"].startswith("much-longer")
    assert out[0].get("reranked") is True
    assert "ce_score" in out[0]


def test_dispatch_falls_back_to_llm_when_model_load_fails(monkeypatch):
    monkeypatch.setenv("V9_RERANKER_BACKEND", "bge-v2-m3")
    _, rr = _fresh_modules()
    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: None)
    monkeypatch.setattr(rr, "_load_ce_reranker", lambda name: None)

    captured = {}

    def _fake_llm(query, candidates, top_k):
        captured["called"] = True
        return candidates[:top_k]

    monkeypatch.setattr(rr, "_rerank_llm", _fake_llm)
    rr._reset_reranker_cache()

    rr.rerank_results(
        "q",
        [{"r": {"content": "x", "project": "p"}, "score": 0.5}],
        top_k=1,
    )
    assert captured.get("called") is True
