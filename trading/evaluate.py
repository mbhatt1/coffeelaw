"""Backtest and evaluation. Costs are charged on turnover; results are net of costs."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def backtest(weights: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = 5.0) -> pd.Series:
    """Daily net returns. Weights at t earn the return from close t to close t+1."""
    fwd = prices.pct_change().shift(-1).fillna(0.0)
    w = weights.reindex(fwd.index).fillna(0.0)
    gross = (w * fwd).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    net = gross - turnover * cost_bps / 1e4
    return net.iloc[:-1]  # last row has no realized next-day return


def metrics(net: pd.Series) -> dict[str, float]:
    n = len(net)
    equity = (1 + net).cumprod()
    cagr = equity.iloc[-1] ** (TRADING_DAYS / n) - 1
    vol = net.std() * np.sqrt(TRADING_DAYS)
    sharpe = net.mean() / net.std() * np.sqrt(TRADING_DAYS) if net.std() > 0 else 0.0
    drawdown = (equity / equity.cummax() - 1).min()
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": drawdown}


def excess_ci(strategy: pd.Series, bench: pd.Series, block: int = 21, reps: int = 2000, seed: int = 0):
    """Annualized mean excess return vs benchmark, with a block-bootstrap 95% interval."""
    diff = (strategy - bench).dropna().to_numpy()
    n = len(diff)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([diff[s : s + block] for s in starts])[:n]
        means[b] = sample.mean()
    point = diff.mean() * TRADING_DAYS
    lo, hi = np.percentile(means, [2.5, 97.5]) * TRADING_DAYS
    return point, lo, hi
