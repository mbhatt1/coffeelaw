"""Price-only features and targets.

Every feature at date t uses prices up to and including the close at t.
The target at date t is the sign of the return from close t to close t+1,
so it is never available when the feature is computed.
"""
from __future__ import annotations

import pandas as pd

FEATURE_NAMES = ["mom1", "mom5", "mom21", "mom63", "vol21"]


def build_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = prices.pct_change()
    return {
        "mom1": prices.pct_change(1),
        "mom5": prices.pct_change(5),
        "mom21": prices.pct_change(21),
        "mom63": prices.pct_change(63),
        "vol21": r.rolling(21).std(),
    }


def next_day_return(prices: pd.DataFrame) -> pd.DataFrame:
    """Return from close t to close t+1, indexed by t. Only used as a label or for backtesting."""
    return prices.pct_change().shift(-1)


def up_target(prices: pd.DataFrame) -> pd.DataFrame:
    fwd = next_day_return(prices)
    return (fwd > 0).astype(float).where(fwd.notna())
