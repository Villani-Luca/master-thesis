"""Long-only top-k holdings from a model's test predictions (THESIS_GUIDE.md Step 4).

Same rules as FinBench's create_long_short_portfolio_history (Evaluation/portfolio/transforms.py,
backtest.py), with short_k = 0. FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.

- Rebalance dates: pd.date_range(start, end, freq=f"{L}B"); each period runs from one date to the next.
- At each period start, the portfolio uses the latest prediction whose last input day is before the
  period start (FinBench raises if it is more than 7 days old).
- The k highest scores are held with equal weights, chosen with np.argsort as in FinBench.

Added for the thesis: every candidate stock of the day is kept, held or not, with its score, rank
and margin to the k-boundary (Step 9), and the overlap between two portfolios (seeds, models).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MAX_PREDICTION_AGE_DAYS = 7  # FinBench _find_relevant_prediction_index


def load_predictions(run_dirs: list[Path]) -> pd.DataFrame:
    """Test predictions of one or more runs (e.g. the test years of a model), from their results pickle.

    One row per stock and last input day: date (last input day), realized (the day the label is
    realized), ticker, score, label.
    """
    frames = []
    for run_dir in run_dirs:
        results_file = next(Path(run_dir).glob("results_sl*_pl*.pkl"))
        with open(results_file, "rb") as f:
            r = pickle.load(f)
        for date, realized, tickers, preds, labels in zip(r["last_date"], r["pred_date"], r["tickers"], r["preds"],
                                                          r["labels"]):
            frames.append(pd.DataFrame({"date": pd.Timestamp(date), "realized": pd.Timestamp(realized), "ticker": tickers,
                                        "score": np.ravel(preds), "label": np.ravel(labels)}))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"], ignore_index=True)


def rebalance_periods(start: str, end: str, pred_len: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """FinBench's get_backtest_iterator with freq = f'{pred_len}B': consecutive pairs of a business-day grid."""
    dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=f"{pred_len}B")
    return [(dates[i], dates[i + 1]) for i in range(len(dates) - 1)]


def topk_holdings(predictions: pd.DataFrame, periods: list[tuple[pd.Timestamp, pd.Timestamp]], k: int) -> pd.DataFrame:
    """Every candidate stock of every period, with its weight in the long-only top-k portfolio.

    Columns: start, end (the holding period), prediction_date (last input day of the prediction
    used), ticker, score, rank (1 = highest score), held, weight (1/k if held, else 0), margin.
    margin is the distance to the k-boundary, always >= 0: for a held stock, its score minus the
    (k+1)-th score (how far it can fall and stay); for another stock, the k-th score minus its
    score (how far it must rise to enter). margin_z is the margin divided by the day's cross-sectional
    standard deviation of the scores, so that margins compare across models with different score scales.
    """
    days = np.array(sorted(predictions["date"].unique()))
    by_day = {d: g for d, g in predictions.groupby("date", sort=False)}
    rows = []
    for start, end in periods:
        candidates = days[days < start]
        if len(candidates) == 0:
            raise ValueError(f"no prediction before the period starting {start.date()}")
        day = candidates[-1]
        if (start - day).days > MAX_PREDICTION_AGE_DAYS:
            raise ValueError(f"the latest prediction before {start.date()} is from {pd.Timestamp(day).date()}")
        g = by_day[day]
        scores = g["score"].to_numpy()
        order = np.argsort(scores)  # FinBench: np.argsort(scores)[-top_k:]
        held = np.zeros(len(g), dtype=bool)
        held[order[-k:]] = True
        rank = np.empty(len(g), dtype=int)
        rank[order[::-1]] = np.arange(1, len(g) + 1)
        kth = scores[order[-k]]
        next_out = scores[order[-k - 1]] if len(g) > k else np.nan
        rows.append(pd.DataFrame({
            "start": start, "end": end, "prediction_date": day, "ticker": g["ticker"].to_numpy(), "score": scores,
            "rank": rank, "held": held, "weight": np.where(held, 1.0 / k, 0.0),
            "margin": np.where(held, scores - next_out, kth - scores),
        }))
        rows[-1]["margin_z"] = rows[-1]["margin"] / scores.std(ddof=1)
    return pd.concat(rows, ignore_index=True)


def holdings_sets(holdings: pd.DataFrame) -> pd.Series:
    """The set of held tickers per period start."""
    return holdings[holdings["held"]].groupby("start")["ticker"].agg(frozenset)


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per common period: Jaccard overlap |A ∩ B| / |A ∪ B| of the held stocks of two portfolios, and the
    one-way turnover between them (the share of the portfolio to trade to go from one to the other)."""
    sa, sb = holdings_sets(a), holdings_sets(b)
    common = sa.index.intersection(sb.index)
    rows = []
    for start in common:
        x, y = sa[start], sb[start]
        rows.append({"start": start, "jaccard": len(x & y) / len(x | y), "turnover": len(x - y) / len(x)})
    return pd.DataFrame(rows)
