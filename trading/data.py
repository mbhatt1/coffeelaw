"""Price data sources. All return a wide DataFrame: index = dates, columns = tickers, values = adjusted close."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_prices_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index().dropna(how="all")


def load_prices_yahoo(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Requires network access to query*.finance.yahoo.com (see README)."""
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.sort_index().dropna(how="all")


def synthetic_prices(
    n_days: int = 1500,
    n_tickers: int = 20,
    seed: int = 0,
    ar: float = 0.0,
    drift: float = 0.0003,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Log-returns with optional AR(1) coefficient `ar`.

    ar=0 is a null world (no predictability). ar>0 plants a known, weak
    predictable component so tests can check the pipeline recovers it.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, vol, (n_days, n_tickers))
    r = np.zeros_like(eps)
    for t in range(n_days):
        prev = r[t - 1] if t > 0 else 0.0
        r[t] = drift + ar * prev + eps[t]
    prices = 100.0 * np.exp(np.cumsum(r, axis=0))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=[f"T{i:02d}" for i in range(n_tickers)])
