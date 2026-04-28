"""v11 D4 LoCoMo — BAAI/bge-reranker-v2-m3 integration tests.

The dispatch layer (`V9_RERANKER_BACKEND` → `_get_reranker`) is exercised
by `tests/test_reranker_backend.py`. This file focuses on the named
BGE-v2-m3 entrypoints introduced in D4:

    * `_get_bge_v2_m3()` — lazy singleton over FlagReranker / CrossEncoder
    * `_bge_rerank_scores(query, passages)` — normalised [0,1] scoring API

Tests are designed to run offline by default — heavy model loads are
mocked. Real model download (~568 MB, fp32 → ~284 MB fp16) only fires
when `RUN_RERANKER_SLOW=1` is set; the slow tests carry `pytest.mark.slow`
and self-skip otherwise.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip every reranker env var so each test starts from defaults."""
    for k in (
        "V9_RERANKER_BACKEND",
        "V9_RERANKER_MODEL",
        "V9_RERANKER_FP16",
        "SKIP_RERANKER",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def _fresh_reranker():
    """Reload reranker module so monkeypatched env vars + cleared singletons
    take effect. Returns the freshly loaded module object."""
    import config as _cfg
    import reranker as _rr

    importlib.reload(_cfg)
    importlib.reload(_rr)
    _rr._reset_reranker_cache()
    _rr._reset_bge_v2_m3_singleton()
    return _rr


class _FakeFlagReranker:
    """Mimic FlagEmbedding.FlagReranker.compute_score."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.calls: list[tuple[list, dict]] = []

    def compute_score(self, pairs, normalize: bool = False, **_kw):
        self.calls.append((list(pairs), {"normalize": normalize}))
        # Deterministic score: longer passage → higher relevance, in [0,1].
        # Without normalize the API would return raw logits — emulate that
        # delta so tests can detect whether normalize=True was honored.
        out = []
        for q, p in pairs:
            base = min(1.0, (len(p) % 50) / 50.0 + 0.05)
            if not normalize:
                # Pretend logits — unbounded. We map [0,1] → [-3,+3].
                base = base * 6.0 - 3.0
            out.append(base)
        return out if len(out) > 1 else out[0]


class _FakeCrossEncoder:
    """Mimic sentence_transformers.CrossEncoder for tests where compute_score
    is missing — exercises the fallback branch in `_bge_rerank_scores`."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls: list[list] = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        # Logits proportional to passage length minus a constant.
        return [float(len(p[1]) - 10) for p in pairs]


# ──────────────────────────────────────────────
# (1) Singleton lazy load
# ──────────────────────────────────────────────


def test_bge_singleton(monkeypatch):
    """First call loads, second call must return the cached instance."""
    rr = _fresh_reranker()
    loads = {"count": 0}

    def _fake_load(name: str):
        loads["count"] += 1
        return _FakeFlagReranker(name)

    monkeypatch.setattr(rr, "_load_flag_reranker", _fake_load)

    a = rr._get_bge_v2_m3()
    b = rr._get_bge_v2_m3()
    assert a is b, "singleton must return the same instance"
    assert loads["count"] == 1, "loader called more than once"


def test_bge_singleton_caches_failure(monkeypatch):
    """Failed load is cached as `False` sentinel — second call must not retry."""
    rr = _fresh_reranker()
    loads = {"count": 0}

    def _fake_load(name: str):
        loads["count"] += 1
        return None  # simulate import / download failure

    monkeypatch.setattr(rr, "_load_flag_reranker", _fake_load)

    assert rr._get_bge_v2_m3() is None
    assert rr._get_bge_v2_m3() is None
    assert loads["count"] == 1, "failed load must be cached, not retried"


# ──────────────────────────────────────────────
# (2) Score shape & normalisation
# ──────────────────────────────────────────────


def test_bge_returns_normalised_scores(monkeypatch):
    """All scores must lie in [0,1] when the FlagReranker path is taken."""
    rr = _fresh_reranker()
    fake = _FakeFlagReranker()
    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: fake)

    scores = rr._bge_rerank_scores(
        "what time is dinner?",
        [
            "Dinner is served at 7pm in the main hall.",
            "The library closes at 9pm.",
            "Breakfast tomorrow at 8am.",
        ],
    )
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores), f"out of range: {scores}"


def test_bge_passes_normalize_true_to_flag_reranker(monkeypatch):
    """The FlagReranker compute_score call must use normalize=True."""
    rr = _fresh_reranker()
    fake = _FakeFlagReranker()
    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: fake)

    rr._bge_rerank_scores("q", ["alpha", "beta"])
    assert len(fake.calls) == 1
    _, kwargs = fake.calls[0]
    assert kwargs["normalize"] is True


def test_bge_respects_input_order(monkeypatch):
    """Output length and order must match input passages."""
    rr = _fresh_reranker()
    fake = _FakeFlagReranker()
    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: fake)

    passages = [
        "first short",
        "second somewhat longer passage about something",
        "third",
        "fourth medium-sized passage",
        "fifth — final",
    ]
    scores = rr._bge_rerank_scores("query", passages)
    assert len(scores) == len(passages) == 5

    # The mocked compute_score keys off `len(p) % 50`, so we can recompute
    # the expected ordering and confirm position-by-position.
    expected = [min(1.0, (len(p) % 50) / 50.0 + 0.05) for p in passages]
    for got, exp in zip(scores, expected):
        assert abs(got - exp) < 1e-6, f"order broken: got={scores}, exp={expected}"


def test_bge_handles_empty_passages():
    """Empty input → empty output, no model load."""
    rr = _fresh_reranker()
    out = rr._bge_rerank_scores("query", [])
    assert out == []


def test_bge_falls_back_to_zero_when_model_unavailable(monkeypatch):
    """Loader returns None → scores are all-zeros of the right length."""
    rr = _fresh_reranker()
    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: None)
    out = rr._bge_rerank_scores("q", ["a", "b", "c"])
    assert out == [0.0, 0.0, 0.0]


def test_bge_cross_encoder_fallback_path(monkeypatch):
    """If the loaded model has no compute_score (CrossEncoder), use predict
    + manual sigmoid — scores must still land in [0,1]."""
    rr = _fresh_reranker()
    fake = _FakeCrossEncoder("BAAI/bge-reranker-v2-m3")
    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: fake)

    scores = rr._bge_rerank_scores("q", ["short", "much longer passage indeed"])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert len(fake.calls) == 1


# ──────────────────────────────────────────────
# (3) Dispatch via env var
# ──────────────────────────────────────────────


def test_dispatch_via_env_var(monkeypatch):
    """V9_RERANKER_BACKEND=bge-v2-m3 must select the BGE model id."""
    monkeypatch.setenv("V9_RERANKER_BACKEND", "bge-v2-m3")
    rr = _fresh_reranker()
    assert rr._resolve_reranker_backend() == "bge-v2-m3"
    assert rr._resolve_reranker_model("bge-v2-m3") == "BAAI/bge-reranker-v2-m3"


def test_default_remains_ce_marco():
    """Empty env → default backend stays ce-marco for backward compat."""
    rr = _fresh_reranker()
    assert rr._resolve_reranker_backend() == "ce-marco"
    assert rr._resolve_reranker_model("ce-marco") == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_off_mode_skips_rerank(monkeypatch):
    """V9_RERANKER_BACKEND=off → rerank_results returns top_k as-is, no model
    load."""
    monkeypatch.setenv("V9_RERANKER_BACKEND", "off")
    rr = _fresh_reranker()

    sentinel_calls = {"count": 0}
    monkeypatch.setattr(
        rr,
        "_get_reranker",
        lambda backend=None: (sentinel_calls.__setitem__("count", sentinel_calls["count"] + 1) or (None, None)),
    )

    results = [
        {"r": {"content": "alpha", "project": "p"}, "score": 0.9},
        {"r": {"content": "beta", "project": "p"}, "score": 0.5},
        {"r": {"content": "gamma", "project": "p"}, "score": 0.1},
    ]
    out = rr.rerank_results("q", results, top_k=2)
    assert [x["score"] for x in out] == [0.9, 0.5]
    assert sentinel_calls["count"] == 0, "off mode must short-circuit before _get_reranker"


def test_dispatch_uses_bge_when_backend_set(monkeypatch):
    """End-to-end: V9_RERANKER_BACKEND=bge-v2-m3 → rerank_results actually
    calls the FlagReranker compute_score path through the dispatcher."""
    monkeypatch.setenv("V9_RERANKER_BACKEND", "bge-v2-m3")
    rr = _fresh_reranker()

    fake = _FakeFlagReranker()
    monkeypatch.setattr(rr, "_load_flag_reranker", lambda name: fake)
    rr._reset_reranker_cache()

    results = [
        {"r": {"content": "short", "project": "p"}, "score": 0.4},
        {"r": {"content": "much-longer-content-here", "project": "p"}, "score": 0.5},
    ]
    out = rr.rerank_results("q", results, top_k=2)

    assert len(fake.calls) == 1, "dispatcher must invoke FlagReranker.compute_score once"
    assert all(item.get("reranked") is True for item in out[:2])


# ──────────────────────────────────────────────
# (4) Layer-wall regression
# ──────────────────────────────────────────────


def _walk_pkg(pkg_dir: Path) -> list[Path]:
    return [p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts]


def _module_imports(py_path: Path) -> set[str]:
    try:
        source = py_path.read_text()
    except (UnicodeDecodeError, OSError):
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                out.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module.split(".")[0])
    return out


def test_layer_wall_intact():
    """Re-run the AST check from test_v11_layer_separation here so a
    BGE wiring change cannot quietly break the wall."""
    core = SRC / "memory_core"
    if not core.is_dir():
        pytest.skip("memory_core/ absent — wall check N/A")

    offenders: list[str] = []
    for py in _walk_pkg(core):
        for top in _module_imports(py):
            if top == "ai_layer":
                offenders.append(str(py.relative_to(SRC)))

    # Also confirm the BGE entrypoints are reachable through the ai_layer shim.
    importlib.invalidate_caches()
    shim = importlib.import_module("ai_layer.reranker")
    assert hasattr(shim, "_get_bge_v2_m3"), "ai_layer.reranker shim missing _get_bge_v2_m3"
    assert hasattr(shim, "_bge_rerank_scores"), "ai_layer.reranker shim missing _bge_rerank_scores"

    assert offenders == [], (
        f"memory_core modules import ai_layer (forbidden by v11 §1): "
        f"{sorted(set(offenders))}"
    )


# ──────────────────────────────────────────────
# (5) Slow / live-model tests (skipped offline)
# ──────────────────────────────────────────────


_SLOW_ENABLED = os.environ.get("RUN_RERANKER_SLOW") == "1"


@pytest.mark.slow
@pytest.mark.skipif(not _SLOW_ENABLED, reason="set RUN_RERANKER_SLOW=1 to download bge-reranker-v2-m3 (~568MB)")
def test_p95_latency_bge_under_200ms():
    """End-to-end timing on 10 query-passage pairs.

    Budget: p95 < 200ms per `_bge_rerank_scores` call on a warm model.
    Cold load (first call) is excluded — we run a 1-pair warmup first.
    Network/download cost is not measured.
    """
    rr = _fresh_reranker()

    # Warmup: trigger the actual model download + first forward pass.
    rr._bge_rerank_scores("warmup", ["initial passage to compile the model graph"])

    pairs_query = "When does the conference start?"
    passages = [
        "The opening keynote is at 9am on Monday.",
        "Lunch is served from 12:30pm to 2pm in the main hall.",
        "Registration desk closes at 6pm on the first day.",
        "The poster session runs Tuesday afternoon.",
        "Coffee breaks are scheduled mid-morning and mid-afternoon.",
        "Please bring your badge to all sessions.",
        "Wi-Fi password is printed on the back of your badge.",
        "Evening reception starts at 7pm on Monday in the courtyard.",
        "Speakers must check in at the AV booth 15 minutes before talk time.",
        "Closing remarks are at 4pm on Wednesday.",
    ]

    timings: list[float] = []
    for _ in range(20):
        t0 = time.perf_counter()
        scores = rr._bge_rerank_scores(pairs_query, passages)
        timings.append((time.perf_counter() - t0) * 1000.0)
        assert len(scores) == 10
        assert all(0.0 <= s <= 1.0 for s in scores)

    timings.sort()
    p95_idx = max(0, int(round(0.95 * len(timings))) - 1)
    p95 = timings[p95_idx]
    assert p95 < 200.0, f"p95 latency {p95:.1f}ms exceeds 200ms budget; samples={timings}"


@pytest.mark.slow
@pytest.mark.skipif(not _SLOW_ENABLED, reason="set RUN_RERANKER_SLOW=1 to download bge-reranker-v2-m3 (~568MB)")
def test_real_bge_orders_relevant_passage_first():
    """Smoke test on the live model: an obviously relevant passage must
    score higher than an obviously irrelevant one."""
    rr = _fresh_reranker()
    scores = rr._bge_rerank_scores(
        "What time is the keynote?",
        [
            "The opening keynote is at 9am on Monday.",
            "Bananas are yellow and grow on tropical trees.",
        ],
    )
    assert scores[0] > scores[1], f"BGE failed obvious ranking: {scores}"
