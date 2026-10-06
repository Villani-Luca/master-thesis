"""Prediction metrics: daily cross-sectional IC and RankIC, and pooled error metrics.

Source: code/finbench/Regression/MASTER/base_model.py (calc_ic, predict). Changes:

- RankIC is the Pearson correlation of the ranks (average ranks for ties), which is what pandas'
  method='spearman' computes, without needing scipy.
- R2 is r2_score(labels, preds). FinBench calls r2_score(preds, labels), with the arguments swapped.
- ICIR and RankICIR (mean / sample std of the daily values, as Qlib) are added.
"""

import numpy as np
import pandas as pd


def daily_ic(pred: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """IC (Pearson) and RankIC (Spearman) of one day's cross-section; NaN pairs are ignored."""
    df = pd.DataFrame({"pred": np.ravel(pred), "label": np.ravel(label)})
    return df["pred"].corr(df["label"]), df["pred"].rank().corr(df["label"].rank())


def r2(label: np.ndarray, pred: np.ndarray) -> float:
    label, pred = np.ravel(label), np.ravel(pred)
    return float(1 - np.sum((label - pred) ** 2) / np.sum((label - label.mean()) ** 2))


def summary(preds: list[np.ndarray], labels: list[np.ndarray]) -> dict[str, float]:
    """Metrics of a split from per-day predictions and labels (same order, one array per day).

    Error metrics pool all stock-days; IC metrics average the daily values.
    """
    pred, label = np.concatenate([np.ravel(p) for p in preds]), np.concatenate([np.ravel(y) for y in labels])
    daily = pd.DataFrame([daily_ic(p, y) for p, y in zip(preds, labels)], columns=["IC", "RankIC"])
    error = pred - label
    return {
        "MSE": float(np.mean(error**2)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": r2(label, pred),
        "IC": float(daily["IC"].mean()),
        "RankIC": float(daily["RankIC"].mean()),
        "ICIR": float(daily["IC"].mean() / daily["IC"].std()),
        "RankICIR": float(daily["RankIC"].mean() / daily["RankIC"].std()),
    }
