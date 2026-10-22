"""Statistics on synthetic data with known answers."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from ceilings.analysis import changepoint
from ceilings.analysis.equivalence import holm, superiority, tost
from ceilings.analysis.gains import bootstrap_ci
from ceilings.analysis.usable_info import usable_information


def _paired(gain, n, rho, rng):
    """Correlated binary outcomes for baseline and arm with a given true gain."""
    p0 = 0.4
    z = rng.random(n)
    base = (z < p0).astype(float)
    arm = np.where(rng.random(n) < rho, base, (rng.random(n) < p0).astype(float))
    lift = rng.random(n) < gain / (1 - p0)
    arm = np.where((arm == 0) & lift, 1.0, arm)
    return arm - base


def test_bootstrap_ci_covers_known_gain():
    rng = np.random.default_rng(0)
    hits = 0
    for rep in range(100):
        d = _paired(0.10, 600, 0.5, rng)
        true_mean = 0.10
        lo, hi = bootstrap_ci(d, seed=rep, n_boot=2000)
        hits += lo <= true_mean <= hi
    assert hits >= 90


def test_tost_power_and_rejection():
    rng = np.random.default_rng(1)
    # n = 850 paired at rho = 0.5 is the corrected planning figure (research R10).
    equiv = np.mean([tost(_paired(0.0, 850, 0.5, rng), 0.05) < 0.05 for _ in range(200)])
    assert equiv >= 0.75
    false_equiv = np.mean([tost(_paired(0.10, 850, 0.5, rng), 0.05) < 0.05 for _ in range(200)])
    assert false_equiv <= 0.05


def test_superiority_detects_gain():
    rng = np.random.default_rng(2)
    assert superiority(_paired(0.10, 600, 0.5, rng)) < 0.01
    assert superiority(np.zeros(10)) == 1.0


def test_holm_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.2]
    assert holm(p) == pytest.approx(list(multipletests(p, method="holm")[1]))


def test_usable_information_bayes_optimal_four_servers():
    """Bayes posterior after probing both bits once (p=0.1): usable ~= I = 1.062 bits."""
    rng = np.random.default_rng(3)
    n, eps = 20_000, 0.1
    theta = rng.integers(0, 4, n)
    obs = np.stack([(theta >> b) & 1 ^ (rng.random(n) < eps) for b in (0, 1)], axis=1)
    lik = np.ones((n, 4))
    for t in range(4):
        for b in (0, 1):
            lik[:, t] *= np.where(((t >> b) & 1) == obs[:, b], 1 - eps, eps)
    post = lik / lik.sum(axis=1, keepdims=True)
    res = usable_information(post, theta, 2.0, n_boot=200)
    assert res["usable_bits"] == pytest.approx(1.062, abs=0.02)


def test_usable_information_one_look_agent():
    """Seeing one answer without its bit index caps usable information at 0.2655 bits."""
    rng = np.random.default_rng(4)
    n, eps = 20_000, 0.1
    theta = rng.integers(0, 4, n)
    idx = rng.integers(0, 2, n)
    o = ((theta >> idx) & 1) ^ (rng.random(n) < eps)
    p1 = np.array([[0.1, 0.5, 0.5, 0.9][t] for t in range(4)])
    lik = np.where(o[:, None] == 1, p1[None, :], 1 - p1[None, :])
    post = lik / lik.sum(axis=1, keepdims=True)
    res = usable_information(post, theta, 2.0, n_boot=200)
    assert res["usable_bits"] <= 0.2655 + 0.02


def test_changepoint_prefers_threshold_on_threshold_data():
    rng = np.random.default_rng(5)
    L = np.repeat(np.arange(0, 17, 2), 200).astype(float)
    p = np.where(L < 8, 0.5, 0.5 + 0.5 / (1 + np.exp(-(L - 10))))
    y = (rng.random(len(L)) < p).astype(float)
    assert changepoint.compare(L, y)["threshold_wins"]


def test_changepoint_prefers_linear_on_linear_data():
    rng = np.random.default_rng(6)
    L = np.repeat(np.arange(0, 17, 2), 200).astype(float)
    p = 1 / (1 + np.exp(-(-2.0 + 0.25 * L)))
    y = (rng.random(len(L)) < p).astype(float)
    assert not changepoint.compare(L, y, chance=0.5)["threshold_wins"]
