"""H7: does accuracy vs message bits look flat-then-rising (threshold) or steadily rising?"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-6


def _sigmoid(z):
    return expit(z)


def _ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_linear(L, y):
    """Logistic in L: p = sigmoid(a + b L)."""
    res = minimize(
        lambda w: -_ll(_sigmoid(w[0] + w[1] * L), y), x0=[0.0, 0.1], method="Nelder-Mead"
    )
    return res.x


def predict_linear(w, L):
    return _sigmoid(w[0] + w[1] * L)


def fit_threshold(L, y, chance: float = 0.5):
    """Flat at `chance` for L < L*, then logistic rise from L*; L* chosen by grid search."""
    best = None
    for star in np.unique(L):

        def nll(w, star=star):
            p = np.where(
                L < star, chance, chance + (1 - chance) * _sigmoid(w[0] + w[1] * (L - star))
            )
            return -_ll(p, y)

        res = minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead")
        if best is None or res.fun < best[0]:
            best = (res.fun, star, res.x)
    return best[1], best[2]


def predict_threshold(params, L, chance: float = 0.5):
    star, w = params
    return np.where(L < star, chance, chance + (1 - chance) * _sigmoid(w[0] + w[1] * (L - star)))


def compare(L, y, chance: float = 0.5, folds: int = 5, seed: int = 0) -> dict:
    """Held-out log-likelihood of the threshold model minus the linear model (positive = threshold)."""
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    fold = np.random.default_rng(seed).permutation(len(y)) % folds
    ll_lin = ll_thr = 0.0
    for f in range(folds):
        tr, te = fold != f, fold == f
        ll_lin += _ll(predict_linear(fit_linear(L[tr], y[tr]), L[te]), y[te])
        ll_thr += _ll(predict_threshold(fit_threshold(L[tr], y[tr], chance), L[te], chance), y[te])
    star, _ = fit_threshold(L, y, chance)
    return {
        "heldout_ll_threshold": ll_thr,
        "heldout_ll_linear": ll_lin,
        "threshold_wins": ll_thr > ll_lin,
        "L_star": float(star),
    }
