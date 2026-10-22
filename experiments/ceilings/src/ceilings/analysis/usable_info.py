"""Usable information = H(Theta) - held-out log-loss of calibrated predictions (paper Sec. VII).

Temperature scaling is cross-fitted (2 folds) so each prediction is calibrated on the other half.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

FLOOR = 1e-6


def _temper(probs: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(probs, FLOOR, 1)) / t
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def log_loss_bits(probs: np.ndarray, truth_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(truth_idx)), truth_idx], FLOOR, 1)
    return float(-np.log2(p).mean())


def fit_temperature(probs: np.ndarray, truth_idx: np.ndarray) -> float:
    res = minimize_scalar(
        lambda t: log_loss_bits(_temper(probs, t), truth_idx), bounds=(0.05, 20.0), method="bounded"
    )
    return float(res.x)


def usable_information(
    probs: np.ndarray, truth_idx: np.ndarray, h_theta: float, seed: int = 0, n_boot: int = 2000
) -> dict:
    """Cross-fitted calibrated estimate with a bootstrap CI over instances."""
    probs = np.asarray(probs, dtype=float)
    truth_idx = np.asarray(truth_idx, dtype=int)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(truth_idx)) % 2
    calibrated = np.empty_like(probs)
    for f in (0, 1):
        fit, apply = folds != f, folds == f
        t = fit_temperature(probs[fit], truth_idx[fit]) if fit.sum() > 1 else 1.0
        calibrated[apply] = _temper(probs[apply], t)
    per_item = -np.log2(np.clip(calibrated[np.arange(len(truth_idx)), truth_idx], FLOOR, 1))
    est = h_theta - per_item.mean()
    idx = rng.integers(0, len(per_item), size=(n_boot, len(per_item)))
    boot = h_theta - per_item[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "usable_bits": float(est),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "raw_logloss": log_loss_bits(probs, truth_idx),
    }
