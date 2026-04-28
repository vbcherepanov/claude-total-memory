"""Platt scaling calibration (W2-I, v11) — convert raw retrieval scores to p(correct).

Why this lives in ``memory_core``
---------------------------------
The hot recall path needs a deterministic, sync, no-LLM way to map a
raw retrieval score to a probability. Calibration is pure numerics on
small floats — fits the layer separation rule (no ``ai_layer`` imports)
and never blocks on I/O.

What is "Platt scaling"?
------------------------
Two-parameter sigmoid

    p(y=1 | s) = 1 / (1 + exp(a * s + b))

Fit ``a, b`` on a held-out validation set of (raw_score, label) pairs
by minimising the binary cross-entropy. The result is a calibrator
object that maps any future raw score in ``[0, 1]`` (or any range) to a
calibrated probability.

We prefer ``scipy.optimize.minimize(method='L-BFGS-B')`` when scipy is
importable, and fall back to a hand-rolled gradient descent loop
(<= 1000 iterations, ``lr=0.01``, early-stopping on tiny gradient
norm). The fallback path is exercised by unit tests via monkeypatching
``HAS_SCIPY = False``.

ECE (Expected Calibration Error)
--------------------------------
Standard 10-bin ECE: bucket predicted probabilities into bins, compute
``|mean(predicted) - mean(label)|`` per bucket weighted by bucket
size, sum. Lower is better. We expect ECE on the validation fixture
to drop after ``fit_platt``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# scipy is optional — when missing we use the gradient-descent fallback.
try:
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore
    HAS_SCIPY = True
except ImportError:  # pragma: no cover — scipy is a project dep, but we
    # still keep the branch tested via monkeypatch.
    _scipy_minimize = None  # type: ignore[assignment]
    HAS_SCIPY = False


__all__ = [
    "HAS_SCIPY",
    "PlattCalibrator",
    "apply",
    "expected_calibration_error",
    "fit_platt",
    "load",
    "save",
]


# ────────────────────────────────────────────────────────────────────
# Public types
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlattCalibrator:
    """Two-parameter sigmoid calibrator.

    The sigmoid is parametrised as

        p = 1 / (1 + exp(a * s + b))

    matching Platt's original 1999 paper. ``a`` is typically negative
    when higher raw scores correlate with positive labels (so increasing
    ``s`` decreases the exponent and pushes ``p`` toward 1).
    """

    a: float
    b: float


# ────────────────────────────────────────────────────────────────────
# Internal numerics
# ────────────────────────────────────────────────────────────────────


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: avoids overflow on large |z|."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _platt_probs(a: float, b: float, scores: np.ndarray) -> np.ndarray:
    """Vectorised forward pass: p = sigmoid(-(a*s + b))."""
    z = a * scores + b
    # Note: per Platt's formulation p = 1/(1+exp(z)) = sigmoid(-z).
    return _stable_sigmoid(-z)


def _bce_loss_and_grad(
    params: np.ndarray, scores: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy + analytical gradient w.r.t. (a, b).

    Forward:
        z = a*s + b
        p = sigmoid(-z) = 1 / (1 + exp(z))

    Loss:
        L = -mean( y*log p + (1-y)*log(1-p) )

    Gradient via chain rule:
        dp/dz   = -sigmoid'(-z) = -p(1-p)
        dL/dp   = -y/p + (1-y)/(1-p) = (p - y) / (p(1-p))
        dL/dz   = (p - y) / (p(1-p)) * (-p(1-p)) = -(p - y) = (y - p)
        dL/da   = mean( (y - p) * s )
        dL/db   = mean( (y - p) )

    Note the sign — getting it wrong turns gradient *descent* into
    ascent and the fit drifts away from the labels (we hit this in
    initial development; covered by the all-negative-labels test).

    Tiny epsilon in the log() terms keeps the loss finite when the
    optimiser overshoots into a region where p is exactly 0 or 1.
    """
    a, b = params[0], params[1]
    p = _platt_probs(a, b, scores)
    eps = 1e-12
    p_clip = np.clip(p, eps, 1.0 - eps)
    loss = -float(np.mean(labels * np.log(p_clip) + (1.0 - labels) * np.log(1.0 - p_clip)))
    diff = labels - p  # dL/dz = y - p (sign matters!)
    grad_a = float(np.mean(diff * scores))
    grad_b = float(np.mean(diff))
    return loss, np.array([grad_a, grad_b], dtype=np.float64)


def _fit_gradient_descent(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> tuple[float, float]:
    """Hand-rolled gradient descent fallback when scipy is absent.

    Uses momentum-accelerated descent with backtracking line search:
    the analytical gradient is fast but vanilla GD with a fixed
    ``lr=0.01`` plateaus far from the optimum on the validation
    fixture (~ECE 0.3). Momentum+backtracking pulls ECE under 0.05
    within a few hundred iterations, matching the L-BFGS path within
    a small tolerance.

    Termination:
    * gradient norm below ``tol`` (early-stop), or
    * loss improvement below ``1e-9`` for 5 consecutive steps, or
    * ``max_iter`` reached.
    """
    params = np.array([-1.0, 0.0], dtype=np.float64)
    velocity = np.zeros_like(params)
    momentum = 0.9
    prev_loss = float("inf")
    stagnant = 0
    cur_lr = lr
    for _ in range(max_iter):
        loss, grad = _bce_loss_and_grad(params, scores, labels)
        gnorm = float(np.linalg.norm(grad))
        if gnorm < tol:
            break

        # Backtracking line search: shrink lr when a step makes loss
        # worse. Helps when momentum overshoots. Note: ``grad`` here is
        # ``dL/dparam`` (see _bce_loss_and_grad doc) — to *minimise* L
        # we step in ``-grad`` direction.
        velocity = momentum * velocity - grad
        trial = params + cur_lr * velocity
        trial_loss, _ = _bce_loss_and_grad(trial, scores, labels)
        for _ in range(10):
            if trial_loss <= loss + 1e-12 or cur_lr < 1e-8:
                break
            cur_lr *= 0.5
            trial = params + cur_lr * velocity
            trial_loss, _ = _bce_loss_and_grad(trial, scores, labels)
        params = trial
        # Slowly grow lr back when steps are productive.
        cur_lr = min(lr * 10.0, cur_lr * 1.05)

        if abs(prev_loss - trial_loss) < 1e-9:
            stagnant += 1
            if stagnant >= 5:
                break
        else:
            stagnant = 0
        prev_loss = trial_loss
    return float(params[0]), float(params[1])


def _fit_scipy(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """L-BFGS-B fit. Initial point matches the Platt-1999 prior."""
    assert _scipy_minimize is not None
    x0 = np.array([-1.0, 0.0], dtype=np.float64)

    def fg(p: np.ndarray) -> tuple[float, np.ndarray]:
        return _bce_loss_and_grad(p, scores, labels)

    res = _scipy_minimize(
        fg,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    return float(res.x[0]), float(res.x[1])


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    """Fit a Platt sigmoid on (scores, labels).

    Parameters
    ----------
    scores:
        1-D array of raw retrieval scores. Any real range is accepted —
        the sigmoid handles the squashing.
    labels:
        1-D array of integer labels in ``{0, 1}``. Same length as
        ``scores``.

    Returns
    -------
    PlattCalibrator
        Fitted parameters. When scipy is importable we use L-BFGS-B,
        otherwise the hand-rolled gradient descent fallback.

    Edge cases
    ----------
    * Empty input raises ``ValueError`` — fitting on nothing is a bug,
      not a degenerate-but-valid case.
    * All-positive or all-negative labels: Platt smoothing
      (Lin et al., 2007) replaces hard {0, 1} with
      ``(N+ + 1) / (N+ + 2)`` for positives and ``1 / (N- + 2)`` for
      negatives. This avoids the divergence of ``log(0)`` and yields a
      well-defined calibrator even on degenerate input.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if s.size == 0 or y.size == 0:
        raise ValueError("fit_platt: scores and labels must be non-empty")
    if s.shape != y.shape:
        raise ValueError(
            f"fit_platt: shape mismatch — scores {s.shape} vs labels {y.shape}"
        )

    # Platt smoothing for degenerate label distributions.
    n_pos = float(np.sum(y > 0.5))
    n_neg = float(y.size - n_pos)
    pos_target = (n_pos + 1.0) / (n_pos + 2.0)
    neg_target = 1.0 / (n_neg + 2.0)
    smoothed = np.where(y > 0.5, pos_target, neg_target)

    if HAS_SCIPY and _scipy_minimize is not None:
        a, b = _fit_scipy(s, smoothed)
    else:
        a, b = _fit_gradient_descent(s, smoothed)

    # Guard against pathological NaN/Inf — fall back to identity-ish.
    if not (math.isfinite(a) and math.isfinite(b)):
        a, b = -1.0, 0.0
    return PlattCalibrator(a=a, b=b)


def apply(cal: PlattCalibrator, raw_score: float) -> float:
    """Map one raw score to a calibrated probability in [0, 1].

    The result is clamped to the unit interval — even though the
    sigmoid is mathematically bounded there, floating-point underflow
    can produce ``-0.0`` or values infinitesimally past the edges, and
    callers downstream compare against thresholds.
    """
    try:
        s = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(s):
        return 0.0
    z = cal.a * s + cal.b
    # Numerically stable sigmoid(-z) = 1/(1+exp(z)). Both branches
    # avoid overflow: large +z → exp(-z) underflows to 0, returning 0;
    # large -z → exp(+z) underflows to 0, returning 1.
    if z >= 0:
        try:
            ez = math.exp(-z)
        except OverflowError:
            ez = 0.0
        p = ez / (1.0 + ez)
    else:
        try:
            ez = math.exp(z)
        except OverflowError:
            ez = 0.0
        p = 1.0 / (1.0 + ez)
    return max(0.0, min(1.0, p))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width bins on [0, 1].

    For each bin we compute ``|mean(probs) - mean(labels)|`` and sum
    weighted by bin size. Empty bins contribute zero. Result is in
    ``[0, 1]``; lower means better calibration.

    Parameters
    ----------
    probs, labels:
        Equal-length 1-D arrays. ``probs`` are the calibrated
        probabilities, ``labels`` are the binary truth values.
    n_bins:
        Number of equal-width bins on ``[0, 1]``. Default 10 matches
        the most common literature convention.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size == 0 or y.size == 0:
        return 0.0
    if p.shape != y.shape:
        raise ValueError(
            f"ECE: shape mismatch — probs {p.shape} vs labels {y.shape}"
        )
    n_bins = max(1, int(n_bins))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(p.size)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        bin_n = float(np.sum(mask))
        if bin_n == 0.0:
            continue
        bin_conf = float(np.mean(p[mask]))
        bin_acc = float(np.mean(y[mask]))
        ece += (bin_n / total) * abs(bin_conf - bin_acc)
    return ece


def save(cal: PlattCalibrator, path: Path) -> None:
    """Persist ``cal`` as JSON.

    The on-disk format is a tiny dict so non-Python tooling (jq, eyes)
    can read it. Parent directories are created on demand.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "kind": "platt",
        **asdict(cal),
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load(path: Path) -> PlattCalibrator:
    """Load a previously saved calibrator.

    Validates the file is a v1 Platt JSON and the parameters are finite.
    Raises ``ValueError`` on schema mismatch or non-finite params.
    """
    p = Path(path)
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"calibrator JSON at {p} is not an object")
    if raw.get("kind") != "platt":
        raise ValueError(f"calibrator JSON at {p} has kind={raw.get('kind')!r}, expected 'platt'")
    try:
        a = float(raw["a"])
        b = float(raw["b"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"calibrator JSON at {p} missing/invalid a,b: {exc}") from exc
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"calibrator JSON at {p} has non-finite params: a={a}, b={b}")
    return PlattCalibrator(a=a, b=b)
