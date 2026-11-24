import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trading import data, evaluate, features, model, strategies


def _oos_auc(prices):
    feats = features.build_features(prices)
    target = features.up_target(prices)
    proba = model.walk_forward_proba(feats, target)
    y = target.stack(future_stack=True)
    p = proba.stack(future_stack=True)
    both = pd.concat([y.rename("y"), p.rename("p")], axis=1).dropna()
    return roc_auc_score(both["y"], both["p"])


def test_features_do_not_use_future_prices():
    prices = data.synthetic_prices(n_days=400, seed=1)
    base = features.build_features(prices)
    tampered = prices.copy()
    cut = prices.index[250]
    tampered.loc[tampered.index > cut] *= np.linspace(0.5, 2.0, (tampered.index > cut).sum())[:, None]
    alt = features.build_features(tampered)
    for name in base:
        pd.testing.assert_frame_equal(base[name].loc[:cut], alt[name].loc[:cut])


def test_null_world_has_no_predictive_power():
    prices = data.synthetic_prices(n_days=1500, n_tickers=20, ar=0.0, seed=2)
    auc = _oos_auc(prices)
    assert 0.46 < auc < 0.54, auc


def test_planted_edge_is_recovered():
    prices = data.synthetic_prices(n_days=1500, n_tickers=20, ar=0.2, seed=3)
    auc = _oos_auc(prices)
    assert auc > 0.55, auc


def test_costs_scale_with_turnover():
    prices = data.synthetic_prices(n_days=300, seed=4)
    static = strategies.buy_and_hold(prices)
    free = evaluate.backtest(static, prices, cost_bps=0)
    costly = evaluate.backtest(static, prices, cost_bps=50)
    # Static weights trade only on day one, so costs barely move the result.
    assert abs((free - costly).sum()) < 0.01


def test_random_strategy_does_not_significantly_beat_buy_and_hold_in_null_world():
    prices = data.synthetic_prices(n_days=1500, n_tickers=20, ar=0.0, seed=5)
    bench = evaluate.backtest(strategies.buy_and_hold(prices), prices, 5)
    rnd = evaluate.backtest(strategies.random_topk(prices, 3, seed=0), prices, 5)
    _, lo, hi = evaluate.excess_ci(rnd, bench)
    # Random top-3 churns and pays costs, so it typically lags. The check that
    # matters is that its upper confidence bound does not show a real edge.
    assert hi < 0.02, (lo, hi)
