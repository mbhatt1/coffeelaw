# trading: direction model and strategy evaluation

Status: pipeline runs end to end on synthetic data. Real-market results are not yet produced (see "Data access").

## What it does

1. **Features** (`features.py`): price-only momentum (1/5/21/63 days) and 21-day volatility. Each feature at date t uses closes up to t only.
2. **Direction model** (`model.py`): logistic regression predicting whether the next day's return is positive, refit every 21 trading days on an expanding window with a one-day purge.
3. **Strategies** (`strategies.py`): top-k equal weight by P(up), threshold on P(up), momentum top-k, random top-k, buy and hold.
4. **Evaluation** (`evaluate.py`): costs on turnover, CAGR, Sharpe, max drawdown, and block-bootstrap 95% intervals for excess return over buy and hold.
5. **Runner** (`run.py`): wires the above together and prints a table.

Tests (`tests/test_pipeline.py`) check that:
- features do not change when future prices are altered (lookahead guard);
- the model has no predictive power in a null world (AUC near 0.5);
- a planted weak AR(1) signal is recovered (AUC > 0.55);
- costs scale with turnover;
- random strategies do not show a significant edge over buy and hold.

## Run

```
python -m pytest -q trading/tests
python -m trading.run --source synthetic
python -m trading.run --source yahoo --tickers SPY,QQQ,IWM,XLF,XLK,XLE --start 2010-01-01
python -m trading.run --source csv --path prices.csv   # wide CSV: date index, one column per ticker
```

On synthetic null data the model and momentum strategies lag buy and hold after costs. That is the expected result when there is no edge, and it is what the harness should report.

## Data access

The sandbox currently blocks Yahoo Finance and GDELT (HTTP 403 from the egress proxy). To run on real data, either:
- allow `query1.finance.yahoo.com`, `query2.finance.yahoo.com`, and `fc.yahoo.com` in the environment's network settings, or
- drop a wide price CSV in the repo and use `--source csv`.

## Not built yet

- **Sentiment.** The design calls for news and social features scored point-in-time, with a lexicon baseline and an LLM scorer. No news source is reachable from here yet, so the model currently uses price features only.
- **Multiple-testing correction.** Deflated Sharpe over the full variant count is not implemented; the variant count should be logged before headline numbers are quoted.
- **Paper broker and risk controls.** Not started. No code path places orders.
- **Stronger models.** Only logistic regression so far. Gradient boosting or sequence models should be added only if they beat logistic regression out of sample.
