"""W1-E NLI Verifier — entailment check for generated answers (v11.0).

After the LLM produces an answer, this module verifies that the answer is
entailed by the retrieved evidence using a local multilingual NLI model
(`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, ~270 MB).

Public API
----------

    verify(answer, evidence, *, batch_size=8) -> VerifyResult
    warmup() -> None

The model is held as a process-wide lazy singleton and reused across calls.
On Apple Silicon the device is auto-detected as ``mps`` if available, with a
``cpu`` fallback. All forwards run under ``torch.no_grad()`` and pairs are
batched in a single call to keep p95 latency under 50 ms for 5 evidence
pieces on an M-series CPU/MPS after warmup.

Aggregation rule
----------------

For each evidence piece, NLI is run as
``premise=evidence_i, hypothesis=answer``. Aggregation picks the single
"best supporting" piece — the one with the highest P(entail) — and uses
its full distribution to decide. That avoids letting per-class maxima
from different (off-topic) pieces dilute a strong supporter:

* ``p_entail``     = max over evidence of P(entail)
* ``p_contradict`` = max over evidence of P(contradict) (any evidence may veto)
* ``p_neutral``    = P(neutral) of the piece that produced ``p_entail``

The decision is:

* ``CONTRADICT`` if ``p_contradict > 0.6`` — any single piece strongly
  contradicting the answer vetoes entailment.
* ``ENTAIL``     if (after the contradict check) ``p_entail`` dominates
  on the supporting piece and is large enough in absolute terms.
* ``NEUTRAL``    otherwise.

Empty evidence short-circuits to ``VerifyResult(NEUTRAL, 0, 1, 0, 0)`` —
nothing to verify against, so we cannot claim entailment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

# Defer torch / transformers imports into the loader: keeps `from
# ai_layer.verifier import VerifyResult` cheap (no model load on import).


__all__ = [
    "NLIDecision",
    "VerifyResult",
    "verify",
    "warmup",
]


_log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Public types
# ────────────────────────────────────────────────────────────────────


class NLIDecision(str, Enum):
    """Verifier verdict over (answer, evidence)."""

    ENTAIL = "entail"
    NEUTRAL = "neutral"
    CONTRADICT = "contradict"


@dataclass(frozen=True)
class VerifyResult:
    decision: NLIDecision
    p_entail: float
    p_neutral: float
    p_contradict: float
    aggregated_from: int  # number of evidence pieces actually scored


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────

# Multilingual NLI model. ~270 MB. Trained on XNLI + multilingual NLI corpora;
# label order is deterministic (entailment=0, neutral=1, contradiction=2)
# per the model card.
_DEFAULT_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# HuggingFace label → our enum index. Read explicitly from id2label after load
# so we never assume label ordering — covered by tests.
_LABEL_ALIASES = {
    "entailment": NLIDecision.ENTAIL,
    "entail": NLIDecision.ENTAIL,
    "neutral": NLIDecision.NEUTRAL,
    "contradiction": NLIDecision.CONTRADICT,
    "contradict": NLIDecision.CONTRADICT,
}

# Default decision thresholds (used when no calibration file is present
# OR when the file is malformed). These match the original v11.0 W1-E
# behaviour so existing tests remain green.
_DEFAULT_CONTRADICT_THRESHOLD = 0.6
_DEFAULT_ENTAIL_MIN = 0.5
_DEFAULT_CONTRADICT_MARGIN = 0.0
_MAX_PAIR_TOKENS = 256  # premise+hypothesis cap; keeps batches small/fast


def _model_id() -> str:
    return (
        os.environ.get("MEMORY_NLI_MODEL", _DEFAULT_MODEL_ID).strip()
        or _DEFAULT_MODEL_ID
    )


# ────────────────────────────────────────────────────────────────────
# Calibration config (W5)
# ────────────────────────────────────────────────────────────────────


_CALIBRATION_PATH_DEFAULT = "~/.claude-memory/nli_calibration.json"


@dataclass(frozen=True)
class _CalibrationConfig:
    model_name: str
    p_entail_threshold: float
    p_contradict_threshold: float
    p_contradict_margin: float
    source: str  # "file" | "default"

    @classmethod
    def defaults(cls) -> "_CalibrationConfig":
        return cls(
            model_name=_DEFAULT_MODEL_ID,
            p_entail_threshold=_DEFAULT_ENTAIL_MIN,
            p_contradict_threshold=_DEFAULT_CONTRADICT_THRESHOLD,
            p_contradict_margin=_DEFAULT_CONTRADICT_MARGIN,
            source="default",
        )


def _calibration_path() -> Path:
    raw = os.environ.get("MEMORY_NLI_CALIBRATION_PATH", _CALIBRATION_PATH_DEFAULT)
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def _load_calibration() -> _CalibrationConfig:
    """Read calibration JSON from disk; fall back to defaults on any error.

    The calibration file is produced by ``scripts/calibrate_nli.py --tune``.
    Schema (only the four keys below are required; others are ignored)::

        {
          "model_name": "...",
          "p_entail_threshold": 0.5,
          "p_contradict_threshold": 0.6,
          "p_contradict_margin": 0.0
        }
    """
    path = _calibration_path()
    if not path.exists():
        _log.info("nli verifier: no calibration file at %s, using defaults", path)
        return _CalibrationConfig.defaults()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — we explicitly want to fall back.
        _log.warning(
            "nli verifier: calibration file %s is invalid (%s); using defaults",
            path, exc,
        )
        return _CalibrationConfig.defaults()
    try:
        cfg = _CalibrationConfig(
            model_name=str(raw.get("model_name", _DEFAULT_MODEL_ID)).strip()
            or _DEFAULT_MODEL_ID,
            p_entail_threshold=float(raw["p_entail_threshold"]),
            p_contradict_threshold=float(raw["p_contradict_threshold"]),
            p_contradict_margin=float(raw.get("p_contradict_margin", 0.0)),
            source="file",
        )
    except (KeyError, TypeError, ValueError) as exc:
        _log.warning(
            "nli verifier: calibration file %s is missing keys (%s); using defaults",
            path, exc,
        )
        return _CalibrationConfig.defaults()

    # Sanity-clamp: any file value outside [0, 1] is treated as malformed
    # and we fall back to defaults rather than break verifier behaviour.
    for name, val in (
        ("p_entail_threshold", cfg.p_entail_threshold),
        ("p_contradict_threshold", cfg.p_contradict_threshold),
        ("p_contradict_margin", cfg.p_contradict_margin),
    ):
        if not (0.0 <= val <= 1.0):
            _log.warning(
                "nli verifier: calibration `%s=%s` out of [0,1]; using defaults",
                name, val,
            )
            return _CalibrationConfig.defaults()
    return cfg


# ────────────────────────────────────────────────────────────────────
# Lazy singleton loader
# ────────────────────────────────────────────────────────────────────


class _NLIModel:
    """Process-wide singleton wrapping tokenizer + model + device."""

    def __init__(self) -> None:
        # Imported here so module import stays cheap.
        import torch  # noqa: WPS433
        from transformers import (  # noqa: WPS433
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        model_id = _model_id()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.eval()

        self._device = self._pick_device(torch)
        self._model.to(self._device)

        # Resolve label index per class from id2label so we are not coupled to
        # any specific label ordering inside the checkpoint.
        id2label = getattr(self._model.config, "id2label", None) or {}
        self._idx = self._resolve_label_indices(id2label)
        self._torch = torch

    @staticmethod
    def _pick_device(torch_mod):
        # Apple Silicon → MPS. Fallback to CPU. CUDA also handled when present.
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return torch_mod.device("mps")
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        return torch_mod.device("cpu")

    @staticmethod
    def _resolve_label_indices(id2label: dict) -> dict[NLIDecision, int]:
        idx: dict[NLIDecision, int] = {}
        for raw_idx, raw_label in id2label.items():
            key = str(raw_label).strip().lower()
            decision = _LABEL_ALIASES.get(key)
            if decision is None:
                continue
            try:
                idx[decision] = int(raw_idx)
            except (TypeError, ValueError):
                continue
        missing = [d for d in NLIDecision if d not in idx]
        if missing:
            raise RuntimeError(
                f"NLI model {_model_id()} is missing label(s): "
                f"{[d.value for d in missing]}. id2label={id2label!r}"
            )
        return idx

    # ─── inference ────────────────────────────────────────────────

    def score(
        self,
        evidence: Sequence[str],
        answer: str,
        batch_size: int,
    ) -> list[tuple[float, float, float]]:
        """Run NLI for every (evidence_i, answer) pair.

        Returns a list of (p_entail, p_neutral, p_contradict) tuples in the
        same order as ``evidence``.
        """
        torch = self._torch
        results: list[tuple[float, float, float]] = []
        if not evidence:
            return results

        e_idx = self._idx[NLIDecision.ENTAIL]
        n_idx = self._idx[NLIDecision.NEUTRAL]
        c_idx = self._idx[NLIDecision.CONTRADICT]

        # Single tokenizer call across the whole batch is far cheaper than
        # tokenizing per-pair, and `batch_size` controls the forward chunk.
        with torch.no_grad():
            for start in range(0, len(evidence), max(1, batch_size)):
                chunk = list(evidence[start : start + batch_size])
                hypotheses = [answer] * len(chunk)

                enc = self._tokenizer(
                    chunk,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=_MAX_PAIR_TOKENS,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}

                logits = self._model(**enc).logits  # (B, 3)
                probs = torch.softmax(logits, dim=-1).detach().to("cpu").tolist()

                for row in probs:
                    results.append(
                        (
                            float(row[e_idx]),
                            float(row[n_idx]),
                            float(row[c_idx]),
                        )
                    )
        return results


_singleton: _NLIModel | None = None
_singleton_lock = threading.Lock()

# Active calibration configuration. Resolved lazily on first use so import
# of this module stays free of disk I/O.
_active_calibration: _CalibrationConfig | None = None


def _get_model() -> _NLIModel:
    global _singleton, _active_calibration
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            # Resolve calibration BEFORE building the model so that, if the
            # operator pinned a model_name in the JSON, we honour it via env.
            cfg = _load_calibration()
            if cfg.source == "file" and not os.environ.get("MEMORY_NLI_MODEL"):
                # Promote the calibrated model name into the env so
                # ``_model_id()`` picks it up. Don't override an explicit
                # operator-set env var.
                os.environ["MEMORY_NLI_MODEL"] = cfg.model_name
            _active_calibration = cfg
            _singleton = _NLIModel()
    return _singleton


def _get_calibration() -> _CalibrationConfig:
    """Return the active calibration, resolving it on first call.

    Public-ish for tests; not part of the documented API.
    """
    global _active_calibration
    if _active_calibration is not None:
        return _active_calibration
    with _singleton_lock:
        if _active_calibration is None:
            _active_calibration = _load_calibration()
    return _active_calibration


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────


def warmup() -> None:
    """Force model load + a tiny forward pass.

    Intended to be called at server startup so the first user-facing
    ``verify()`` call does not pay tokenizer/model download cost.
    """
    model = _get_model()
    # Tiny dummy pair: enough to JIT kernels on the chosen device.
    model.score(evidence=["warmup"], answer="warmup", batch_size=1)


def _decide(
    p_entail: float,
    p_neutral: float,
    p_contradict: float,
    *,
    calibration: _CalibrationConfig | None = None,
) -> NLIDecision:
    """Decide given the *aggregated* probabilities.

    ``p_entail`` and ``p_neutral`` are taken from the single piece that
    most supports the answer (the one with max P(entail)); ``p_contradict``
    is the strongest contradiction across ALL evidence (so an off-topic
    contradiction veto still applies).

    Two profiles:

    * **Default (no `calibration=` argument)** — matches v11.0 W1-E
      behaviour exactly: strict ``p_contradict > 0.6`` veto, no margin.
      This is the path the legacy test suite exercises.
    * **Calibrated (`calibration` provided)** — uses W5 thresholds:
      ``p_contradict >= τ_c AND (p_contradict - p_entail) >= margin``.
      The margin stops moderate-confidence "contradicts" from vetoing
      answers the model also weakly supports — the dominant false-veto
      failure mode on dialogue evidence.
    """
    if calibration is None:
        # Legacy path. Strict `>` so `p_c == 0.6` does not flip.
        if p_contradict > _DEFAULT_CONTRADICT_THRESHOLD:
            return NLIDecision.CONTRADICT
        if (
            p_entail >= _DEFAULT_ENTAIL_MIN
            and p_entail >= p_neutral
            and p_entail >= p_contradict
        ):
            return NLIDecision.ENTAIL
        return NLIDecision.NEUTRAL

    p_c_threshold = calibration.p_contradict_threshold
    p_e_threshold = calibration.p_entail_threshold
    margin = calibration.p_contradict_margin
    if (
        p_contradict >= p_c_threshold
        and (p_contradict - p_entail) >= margin
    ):
        return NLIDecision.CONTRADICT
    if (
        p_entail >= p_e_threshold
        and p_entail >= p_neutral
        and p_entail >= p_contradict
    ):
        return NLIDecision.ENTAIL
    return NLIDecision.NEUTRAL


def verify(
    answer: str,
    evidence: list[str],
    *,
    batch_size: int = 8,
) -> VerifyResult:
    """Verify whether ``answer`` is entailed by any of ``evidence``.

    Empty / whitespace-only inputs short-circuit without loading the model:
    no evidence ⇒ NEUTRAL with ``p_neutral=1.0``.
    """
    if not evidence:
        return VerifyResult(
            decision=NLIDecision.NEUTRAL,
            p_entail=0.0,
            p_neutral=1.0,
            p_contradict=0.0,
            aggregated_from=0,
        )

    # Drop blank evidence pieces; if everything is blank, treat as empty.
    cleaned = [e for e in evidence if isinstance(e, str) and e.strip()]
    if not cleaned:
        return VerifyResult(
            decision=NLIDecision.NEUTRAL,
            p_entail=0.0,
            p_neutral=1.0,
            p_contradict=0.0,
            aggregated_from=0,
        )

    if not isinstance(answer, str) or not answer.strip():
        # No claim to verify ⇒ neutral, regardless of evidence.
        return VerifyResult(
            decision=NLIDecision.NEUTRAL,
            p_entail=0.0,
            p_neutral=1.0,
            p_contradict=0.0,
            aggregated_from=0,
        )

    bs = batch_size if batch_size and batch_size > 0 else 8

    model = _get_model()
    scored = model.score(cleaned, answer, batch_size=bs)

    # Best-supporting piece: the one whose P(entail) is highest. Use its
    # full per-class distribution for the decision so off-topic neutrals
    # do not crowd out a true supporter.
    best_idx = max(range(len(scored)), key=lambda i: scored[i][0])
    p_entail, p_neutral_for_best, _p_contradict_for_best = scored[best_idx]

    # Contradiction is a global veto — any evidence may rule out entailment,
    # even if a different piece appears to support it.
    p_contradict_global = max(row[2] for row in scored)

    return VerifyResult(
        decision=_decide(
            p_entail,
            p_neutral_for_best,
            p_contradict_global,
            calibration=_get_calibration(),
        ),
        p_entail=float(p_entail),
        p_neutral=float(p_neutral_for_best),
        p_contradict=float(p_contradict_global),
        aggregated_from=len(cleaned),
    )
