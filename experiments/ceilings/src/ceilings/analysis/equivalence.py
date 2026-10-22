"""Paired superiority and equivalence (TOST) tests with Holm correction (research R10)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _mean_se(diffs: np.ndarray) -> tuple[float, float, int]:
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        raise ValueError("need at least two paired differences")
    return float(diffs.mean()), float(diffs.std(ddof=1) / np.sqrt(n)), n - 1


def superiority(diffs) -> float:
    """One-sided p-value for mean paired difference > 0."""
    d, se, df = _mean_se(diffs)
    if se == 0:
        return 0.0 if d > 0 else 1.0
    return float(1 - stats.t.cdf(d / se, df))


def tost(diffs, margin: float = 0.05) -> float:
    """Two one-sided tests: p-value for |mean difference| < margin (small p = equivalent)."""
    d, se, df = _mean_se(diffs)
    if se == 0:
        return 0.0 if abs(d) < margin else 1.0
    p_lower = 1 - stats.t.cdf((d + margin) / se, df)
    p_upper = stats.t.cdf((d - margin) / se, df)
    return float(max(p_lower, p_upper))


def holm(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    return [float(p) for p in multipletests(pvals, method="holm")[1]]
