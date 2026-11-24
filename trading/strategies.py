"""Strategies map a daily score (dates x tickers) to target weights.

Weights at date t are decided using information available at the close of t.
Weights are not normalized to 1: anything not invested is cash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def topk_equal(score: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    rank = score.rank(axis=1, ascending=False, method="first")
    w = (rank <= k).astype(float) / k
    return w.where(score.notna(), 0.0)


def threshold(score: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    n = score.shape[1]
    return (score > thr).astype(float).fillna(0.0) / n


def buy_and_hold(prices: pd.DataFrame) -> pd.DataFrame:
    n = prices.shape[1]
    return pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)


def random_topk(prices: pd.DataFrame, k: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    score = pd.DataFrame(rng.random(prices.shape), index=prices.index, columns=prices.columns)
    return topk_equal(score, k)
