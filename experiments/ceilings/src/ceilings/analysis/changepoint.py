"""H7: does accuracy vs message bits look flat-then-rising (threshold) or steadily rising?

Cross-validation folds keep each instance's runs together (the same instances repeat across L),
and fitted probabilities are clipped to [0.01, 0.99] so one held-out miss cannot dominate.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-6
CLIP = (0.01, 0.99)


def _sigmoid(z):
    return expit(z)


def _ll(p, y):
    p = np.clip(p, *CLIP)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_linear(L, y):
    """Logistic in L: p = sigmoid(a + b L)."""
    res = minimize(lambda w: -_ll(_sigmoid(w[0] + w[1] * L), y), x0=[0.0, 0.1],
                   method="Nelder-Mead")
    return res.x


def predict_linear(w, L):
    return _sigmoid(w[0] + w[1] * L)


def fit_threshold(L, y, chance: float = 0.5):
    """Flat at `chance` for L < L*, then a logistic rise from L*; L* chosen by grid search.

    L* ranges over the tested budgets above the smallest one, so the model always asserts a flat
    region (otherwise it would contain the linear model and could never lose to it)."""
    best = None
    levels = np.unique(L)
    for star in (levels[1:] if len(levels) > 1 else levels):
        def nll(w, star=star):
            rise = chance + (1 - chance) * _sigmoid(w[0] + w[1] * (L - star))
            return -_ll(np.where(L < star, chance, rise), y)

        res = minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead")
        if best is None or res.fun < best[0]:
            best = (res.fun, star, res.x)
    return best[1], best[2]


def predict_threshold(params, L, chance: float = 0.5):
    star, w = params
    return np.where(L < star, chance, chance + (1 - chance) * _sigmoid(w[0] + w[1] * (L - star)))


def compare(L, y, chance: float = 0.5, folds: int = 5, seed: int = 0, groups=None) -> dict:
    """Held-out log-likelihood of the threshold model minus the linear one (positive = threshold).

    `groups` (e.g. instance IDs) keeps all of one group's rows in the same fold.
    """
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is not None:
        uniq = np.unique(groups)
        k = min(folds, len(uniq))
        assign = dict(zip(uniq, rng.permutation(len(uniq)) % k, strict=True))
        fold = np.array([assign[g] for g in groups])
    else:
        k = folds
        fold = rng.permutation(len(y)) % k
    ll_lin = ll_thr = 0.0
    for f in range(k):
        tr, te = fold != f, fold == f
        if not te.any() or not tr.any():
            continue
        ll_lin += _ll(predict_linear(fit_linear(L[tr], y[tr]), L[te]), y[te])
        ll_thr += _ll(predict_threshold(fit_threshold(L[tr], y[tr], chance), L[te], chance),
                      y[te])
    star, _ = fit_threshold(L, y, chance)
    return {"heldout_ll_threshold": ll_thr, "heldout_ll_linear": ll_lin,
            "threshold_wins": ll_thr > ll_lin, "L_star": float(star)}


def slope_pvalue(L, y) -> float:
    """One-sided p-value that accuracy increases with L (Spearman rank correlation; robust to
    separation, unlike a logistic Wald test)."""
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(np.unique(L)) < 2 or len(np.unique(y)) < 2:
        return 1.0
    res = stats.spearmanr(L, y)
    r = float(res.statistic)
    p_two = float(res.pvalue)
    return p_two / 2 if r > 0 else 1 - p_two / 2
