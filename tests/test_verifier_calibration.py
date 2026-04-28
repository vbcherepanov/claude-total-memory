"""W5 — NLI verifier calibration regression tests.

Verifies three properties of the calibration pipeline:

1. The calibration fixture exists, is well-formed, and is deterministic
   (rebuilding it with the pinned seed produces an identical md5).

2. The chosen calibration thresholds, applied to the production model's
   precomputed probabilities on the held-out test split, keep
   ``false_contradict_rate ≤ 0.15`` AND ``contradict_recall ≥ 0.80`` AND
   ``entail_recall ≥ 0.75`` — the explicit acceptance criteria from the
   W5 spec.

3. ``ai_layer.verifier._decide`` consumes the calibration JSON and applies
   the calibrated thresholds correctly, while still falling back to v11.0
   defaults when no calibration file is present.

The probability tensor for the production model is regenerated lazily —
the model load takes ~5s on MPS / ~30s on CPU. To keep CI cheap, the test
caches probabilities per-fixture-md5 inside the test session and skips
the heavy path entirely when ``SKIP_NLI=1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

# Standard repo-relative imports.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import calibrate_nli  # noqa: E402
from ai_layer import verifier  # noqa: E402
from ai_layer.verifier import (  # noqa: E402
    NLIDecision,
    _CalibrationConfig,
    _decide,
    _load_calibration,
)


_FIXTURE = _REPO / "tests" / "fixtures" / "nli_calibration_set.json"
_CALIBRATION_JSON = Path(os.path.expanduser("~/.claude-memory/nli_calibration.json"))
_PRODUCTION_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


SKIP_REASON = "NLI model load disabled (set SKIP_NLI=0 to enable)"
_skip_if_disabled = pytest.mark.skipif(
    os.environ.get("SKIP_NLI") == "1",
    reason=SKIP_REASON,
)


# ────────────────────────────────────────────────────────────────────
# Fixture invariants — fast, no model load
# ────────────────────────────────────────────────────────────────────


def test_fixture_exists_and_is_well_formed() -> None:
    assert _FIXTURE.exists(), (
        "calibration fixture missing — run "
        "`python scripts/calibrate_nli.py --build-fixture`"
    )
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == 1
    assert payload.get("seed") == 42
    triples = payload.get("triples", [])
    assert len(triples) >= 300, f"fixture too small: {len(triples)}"
    labels = {t["gold_label"] for t in triples}
    assert labels == {"entail", "neutral", "contradict"}


def test_fixture_is_deterministic() -> None:
    """Re-building with the same seed must produce an identical md5."""
    triples_a, stats_a = calibrate_nli.build_fixture(
        seed=42, target_entail=120, target_contradict=120, target_neutral=80,
    )
    triples_b, stats_b = calibrate_nli.build_fixture(
        seed=42, target_entail=120, target_contradict=120, target_neutral=80,
    )
    assert triples_a == triples_b
    assert stats_a.to_dict() == stats_b.to_dict()


def test_calibration_json_is_well_formed() -> None:
    assert _CALIBRATION_JSON.exists(), (
        "calibration config missing — run "
        "`python scripts/calibrate_nli.py --tune ...`"
    )
    payload = json.loads(_CALIBRATION_JSON.read_text(encoding="utf-8"))
    for key in (
        "model_name",
        "p_entail_threshold",
        "p_contradict_threshold",
        "p_contradict_margin",
        "calibrated_at",
        "calibration_set_md5",
    ):
        assert key in payload, f"calibration json missing `{key}`"
    for k in ("p_entail_threshold", "p_contradict_threshold", "p_contradict_margin"):
        v = payload[k]
        assert isinstance(v, (int, float))
        assert 0.0 <= float(v) <= 1.0


def test_calibration_set_md5_matches_fixture() -> None:
    """The MD5 stored in the calibration JSON must match the fixture on disk."""
    cal = json.loads(_CALIBRATION_JSON.read_text(encoding="utf-8"))
    triples, stats, seed, md5 = calibrate_nli.load_fixture(_FIXTURE)
    assert cal["calibration_set_md5"] == md5, (
        "calibration JSON was fitted against a different fixture md5 — re-run "
        "`scripts/calibrate_nli.py --tune`"
    )


# ────────────────────────────────────────────────────────────────────
# Decision-rule unit tests — pure function, no model load
# ────────────────────────────────────────────────────────────────────


def test_decide_default_matches_legacy_thresholds() -> None:
    """Without a calibration argument the decision rule must match v11.0 W1-E."""
    # contradict above 0.6 strict-greater wins regardless of entail.
    assert _decide(0.9, 0.0, 0.61) is NLIDecision.CONTRADICT
    # contradict at the boundary (0.6) does NOT flip — strict > preserved.
    assert _decide(0.7, 0.1, 0.6) is NLIDecision.ENTAIL
    # entail dominates.
    assert _decide(0.8, 0.1, 0.1) is NLIDecision.ENTAIL
    # weak entail with neutral majority.
    assert _decide(0.2, 0.7, 0.1) is NLIDecision.NEUTRAL


def test_decide_uses_calibrated_thresholds() -> None:
    cfg = _CalibrationConfig(
        model_name=_PRODUCTION_MODEL,
        p_entail_threshold=0.65,
        p_contradict_threshold=0.4,
        p_contradict_margin=0.0,
        source="file",
    )
    # p_contradict at the calibrated threshold (0.4) DOES flip (>=).
    assert _decide(0.05, 0.05, 0.4, calibration=cfg) is NLIDecision.CONTRADICT
    # Weak contradict is veto'd by the calibrated lower threshold.
    assert _decide(0.20, 0.30, 0.50, calibration=cfg) is NLIDecision.CONTRADICT
    # Strong entail above the new τ_entail still wins when contradict is low.
    assert _decide(0.70, 0.20, 0.10, calibration=cfg) is NLIDecision.ENTAIL
    # Entail just below τ_entail (0.65) stays NEUTRAL.
    assert _decide(0.60, 0.30, 0.10, calibration=cfg) is NLIDecision.NEUTRAL


def test_decide_calibrated_with_margin_blocks_close_calls() -> None:
    """Margin must prevent contradict-veto when entail is comparably strong."""
    cfg = _CalibrationConfig(
        model_name=_PRODUCTION_MODEL,
        p_entail_threshold=0.5,
        p_contradict_threshold=0.5,
        p_contradict_margin=0.30,
        source="file",
    )
    # p_contradict 0.55 vs p_entail 0.40 → margin = 0.15 < 0.30 → no veto.
    assert _decide(0.40, 0.05, 0.55, calibration=cfg) is NLIDecision.NEUTRAL
    # p_contradict 0.85 vs p_entail 0.05 → margin = 0.80 ≥ 0.30 → veto.
    assert _decide(0.05, 0.10, 0.85, calibration=cfg) is NLIDecision.CONTRADICT


def test_load_calibration_from_disk_round_trips() -> None:
    cfg = _load_calibration()
    if not _CALIBRATION_JSON.exists():
        assert cfg.source == "default"
        return
    raw = json.loads(_CALIBRATION_JSON.read_text(encoding="utf-8"))
    assert cfg.source == "file"
    assert cfg.model_name == raw["model_name"]
    assert cfg.p_entail_threshold == pytest.approx(raw["p_entail_threshold"])
    assert cfg.p_contradict_threshold == pytest.approx(raw["p_contradict_threshold"])
    assert cfg.p_contradict_margin == pytest.approx(raw["p_contradict_margin"])


def test_load_calibration_falls_back_when_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_NLI_CALIBRATION_PATH", str(tmp_path / "nope.json"))
    cfg = _load_calibration()
    assert cfg.source == "default"
    assert cfg.p_contradict_threshold == 0.6
    assert cfg.p_entail_threshold == 0.5
    assert cfg.p_contradict_margin == 0.0


def test_load_calibration_falls_back_on_garbage(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{this is not json", encoding="utf-8")
    monkeypatch.setenv("MEMORY_NLI_CALIBRATION_PATH", str(bad))
    cfg = _load_calibration()
    assert cfg.source == "default"


def test_load_calibration_falls_back_on_out_of_range(tmp_path, monkeypatch) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({
            "model_name": _PRODUCTION_MODEL,
            "p_entail_threshold": 1.5,
            "p_contradict_threshold": 0.5,
            "p_contradict_margin": 0.0,
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_NLI_CALIBRATION_PATH", str(bad))
    cfg = _load_calibration()
    assert cfg.source == "default"


# ────────────────────────────────────────────────────────────────────
# End-to-end regression — actually run the production NLI model on the
# fixture and assert the W5 acceptance criteria on the test split.
# ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _scored_test_metrics():
    """Score the production model on the fixture and return EvalReport on test."""
    if os.environ.get("SKIP_NLI") == "1":
        pytest.skip(SKIP_REASON)
    if not _FIXTURE.exists() or not _CALIBRATION_JSON.exists():
        pytest.skip("calibration artefacts missing")

    triples, _stats, _seed, _md5 = calibrate_nli.load_fixture(_FIXTURE)
    _train_idx, test_idx = calibrate_nli._split_train_test(
        triples, seed=42, test_frac=0.30,
    )

    # Per-fixture probability cache to keep re-runs fast.
    fixture_md5 = hashlib.md5(_FIXTURE.read_bytes()).hexdigest()
    cache = (
        Path(os.path.expanduser("~/.cache/claude-memory"))
        / f"nli_probs_{fixture_md5}.json"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        probs = [tuple(row) for row in json.loads(cache.read_text(encoding="utf-8"))]
    else:
        try:
            probs = calibrate_nli._score_with_model(
                _PRODUCTION_MODEL, triples, batch_size=16,
            )
        except Exception as exc:  # noqa: BLE001 — model unavailable on CI.
            pytest.skip(f"NLI model unavailable: {exc!r}")
        cache.write_text(json.dumps(probs), encoding="utf-8")

    cal = json.loads(_CALIBRATION_JSON.read_text(encoding="utf-8"))
    rep = calibrate_nli._evaluate(
        triples,
        probs,
        test_idx,
        p_entail_threshold=cal["p_entail_threshold"],
        p_contradict_threshold=cal["p_contradict_threshold"],
        p_contradict_margin=cal["p_contradict_margin"],
    )
    return rep


@_skip_if_disabled
def test_calibrated_test_split_meets_w5_acceptance(_scored_test_metrics) -> None:
    rep = _scored_test_metrics
    # Print so `pytest -s` shows the actual numbers.
    print(
        f"\n[W5 calibration] test n={rep.n} "
        f"fc-rate={rep.false_contradict_rate:.3f} "
        f"contradict-recall={rep.contradict_recall:.3f} "
        f"entail-recall={rep.entail_recall:.3f} "
        f"balanced-acc={rep.balanced_accuracy:.3f}"
    )
    # The W5 spec: false-contradict ≤ 15% on the held-out half.
    assert rep.false_contradict_rate <= 0.15, (
        f"false_contradict_rate {rep.false_contradict_rate:.3f} > 0.15 — "
        "calibration regressed"
    )
    # And true-contradict recall ≥ 80%.
    assert rep.contradict_recall >= 0.80, (
        f"contradict_recall {rep.contradict_recall:.3f} < 0.80 — "
        "calibration weakened the veto signal"
    )
    # And true-entail recall ≥ 75%.
    assert rep.entail_recall >= 0.75, (
        f"entail_recall {rep.entail_recall:.3f} < 0.75 — "
        "calibration over-tightened the entail threshold"
    )


# ────────────────────────────────────────────────────────────────────
# Verifier integration — confirm the loaded calibration is actually
# what `verify()` ends up using under the hood.
# ────────────────────────────────────────────────────────────────────


def test_active_calibration_picked_up_by_verifier_module(monkeypatch) -> None:
    """`_get_calibration()` must reflect the on-disk calibration file."""
    if not _CALIBRATION_JSON.exists():
        pytest.skip("no calibration JSON on disk")

    # Reset the singleton so we re-read the file.
    monkeypatch.setattr(verifier, "_active_calibration", None, raising=False)
    monkeypatch.setattr(verifier, "_singleton", None, raising=False)
    cfg = verifier._get_calibration()
    raw = json.loads(_CALIBRATION_JSON.read_text(encoding="utf-8"))
    assert cfg.source == "file"
    assert cfg.p_entail_threshold == pytest.approx(raw["p_entail_threshold"])
    assert cfg.p_contradict_threshold == pytest.approx(raw["p_contradict_threshold"])
    assert cfg.p_contradict_margin == pytest.approx(raw["p_contradict_margin"])
