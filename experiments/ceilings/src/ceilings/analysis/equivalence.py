"""Paired superiority and equivalence tests with Holm correction (research R10).

Outcomes are paired per instance: correctness under an arm and under its reference. For binary
outcomes (one replicate) the tests are exact or score-based, so they stay valid at small n and when
every difference is zero:

- superiority: exact one-sided McNemar test on the discordant pairs;
- equivalence (TOST): the paired-difference interval of Newcombe (1998, method 10) at level
  1 - 2*alpha, inverted to a p-value (the smallest alpha at which the interval lies inside
  (-margin, +margin)).

Non-binary outcomes (replicates averaged) fall back to paired t-tests, with the standard error
floored so a sample of identical differences cannot produce p = 0.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _as_pairs(arm, ref) -> tuple[np.ndarray, np.ndarray]:
    arm = np.asarray(arm, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if arm.shape != ref.shape or arm.ndim != 1:
        raise ValueError("arm and reference outcomes must be paired 1-d arrays")
    if len(arm) < 2:
        raise ValueError("need at least two paired instances")
    return arm, ref


def _binary(arm: np.ndarray, ref: np.ndarray) -> bool:
    return bool(np.isin(arm, (0.0, 1.0)).all() and np.isin(ref, (0.0, 1.0)).all())


def superiority(arm, ref) -> float:
    """One-sided p-value that the arm's accuracy exceeds the reference's."""
    arm, ref = _as_pairs(arm, ref)
    if _binary(arm, ref):
        wins = int(((arm == 1) & (ref == 0)).sum())
        losses = int(((arm == 0) & (ref == 1)).sum())
        if wins + losses == 0:
            return 1.0
        return float(stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)
    d = arm - ref
    se = max(float(d.std(ddof=1) / math.sqrt(len(d))), 0.5 / len(d))
    return float(1 - stats.t.cdf(d.mean() / se, len(d) - 1))


def _wilson(x: int, n: int, z: float) -> tuple[float, float]:
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def newcombe_paired_ci(arm, ref, level: float) -> tuple[float, float, float]:
    """(difference, lower, upper) for paired binary data, Newcombe (1998) method 10."""
    arm, ref = _as_pairs(arm, ref)
    n = len(arm)
    e = int(((arm == 1) & (ref == 1)).sum())
    f = int(((arm == 1) & (ref == 0)).sum())
    g = int(((arm == 0) & (ref == 1)).sum())
    h = n - e - f - g
    p1, p2 = (e + f) / n, (e + g) / n
    z = stats.norm.ppf(1 - (1 - level) / 2)
    l1, u1 = _wilson(e + f, n, z)
    l2, u2 = _wilson(e + g, n, z)
    num = e * h - f * g
    den = math.sqrt((e + f) * (g + h) * (e + g) * (f + h))
    phi = 0.0 if den == 0 else (max(num - n / 2, 0) if num > 0 else num) / den
    d = p1 - p2
    lower = d - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)
                              + (u2 - p2) ** 2))
    upper = d + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)
                              + (p2 - l2) ** 2))
    return d, lower, upper


def tost(arm, ref, margin: float = 0.05) -> float:
    """Equivalence p-value: small means the accuracy difference lies within +/- margin."""
    arm, ref = _as_pairs(arm, ref)
    if _binary(arm, ref):
        def inside(alpha: float) -> bool:
            _, lo, hi = newcombe_paired_ci(arm, ref, 1 - 2 * alpha)
            return -margin < lo and hi < margin

        if not inside(0.5 - 1e-9):
            return 1.0
        lo_a, hi_a = 1e-12, 0.5 - 1e-9  # inside(hi_a) holds; find the smallest alpha that holds
        if inside(lo_a):
            return lo_a
        for _ in range(60):
            mid = (lo_a + hi_a) / 2
            lo_a, hi_a = (lo_a, mid) if inside(mid) else (mid, hi_a)
        return float(hi_a)
    d = arm - ref
    se = max(float(d.std(ddof=1) / math.sqrt(len(d))), 0.5 / len(d))
    df = len(d) - 1
    p_lower = 1 - stats.t.cdf((d.mean() + margin) / se, df)
    p_upper = stats.t.cdf((d.mean() - margin) / se, df)
    return float(max(p_lower, p_upper))


def holm(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    return [float(p) for p in multipletests(pvals, method="holm")[1]]
