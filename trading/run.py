"""End-to-end run: data -> features -> walk-forward P(up) -> candidate strategies -> evaluation.

Usage:
  python -m trading.run --source synthetic
  python -m trading.run --source yahoo --tickers SPY,QQQ,IWM,XLF,XLK,XLE --start 2010-01-01
  python -m trading.run --source csv --path prices.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from trading import data, evaluate, features, model, strategies


def run(prices: pd.DataFrame, cost_bps: float = 5.0, k: int = 3) -> pd.DataFrame:
    feats = features.build_features(prices)
    target = features.up_target(prices)
    proba = model.walk_forward_proba(feats, target)
    first = proba.dropna(how="all").index[0]
    prices_oos = prices.loc[first:]
    proba_oos = proba.loc[first:]

    candidates = {
        f"model top{k}": strategies.topk_equal(proba_oos, k),
        "model P(up)>0.5": strategies.threshold(proba_oos, 0.5),
        f"momentum21 top{k}": strategies.topk_equal(feats["mom21"].loc[first:], k),
        f"random top{k}": strategies.random_topk(prices_oos, k, seed=0),
        "buy & hold": strategies.buy_and_hold(prices_oos),
    }
    bench = evaluate.backtest(candidates["buy & hold"], prices_oos, cost_bps)

    rows = []
    for name, w in candidates.items():
        net = evaluate.backtest(w, prices_oos, cost_bps)
        m = evaluate.metrics(net)
        point, lo, hi = evaluate.excess_ci(net, bench)
        rows.append({"strategy": name, **m, "excess_vs_bh": point, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).set_index("strategy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "yahoo", "csv"], default="synthetic")
    ap.add_argument("--tickers", default="SPY,QQQ,IWM,XLF,XLK,XLE")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--path", default=None)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    args = ap.parse_args()

    if args.source == "synthetic":
        prices = data.synthetic_prices(ar=0.0)
    elif args.source == "yahoo":
        prices = data.load_prices_yahoo(args.tickers.split(","), args.start)
    else:
        prices = data.load_prices_csv(args.path)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print(f"{prices.shape[1]} tickers, {len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")
    print(run(prices, cost_bps=args.cost_bps).to_string())


if __name__ == "__main__":
    main()
