"""Walk-forward direction model: P(next-day return > 0) per (date, ticker).

Refit every `step` trading days on an expanding window. Training rows whose
label would overlap the first test date are excluded (purge of one day).
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def walk_forward_proba(
    feats: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    min_train: int = 252,
    step: int = 21,
    purge: int = 1,
    C: float = 1.0,
) -> pd.DataFrame:
    """Returns a wide DataFrame (dates x tickers) of out-of-sample P(up)."""
    X = pd.concat({n: f.stack(future_stack=True) for n, f in feats.items()}, axis=1)
    y = target.stack(future_stack=True).rename("y")
    data = X.join(y, how="inner").dropna()
    dates = sorted(data.index.get_level_values(0).unique())
    all_dates = sorted(X.dropna().index.get_level_values(0).unique())

    preds = []
    for i in range(min_train, len(dates), step):
        test_start = dates[i]
        train = data[data.index.get_level_values(0) < dates[i - purge]]
        test_dates = [d for d in all_dates if d >= test_start][:step]
        test = X.dropna()
        test = test[test.index.get_level_values(0).isin(test_dates)]
        if test.empty:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        model.fit(train[list(feats)], train["y"])
        p = model.predict_proba(test[list(feats)])[:, 1]
        preds.append(pd.Series(p, index=test.index))

    if not preds:
        return pd.DataFrame(index=target.index, columns=target.columns, dtype=float)
    proba = pd.concat(preds).unstack()
    return proba.reindex(index=target.index, columns=target.columns)
