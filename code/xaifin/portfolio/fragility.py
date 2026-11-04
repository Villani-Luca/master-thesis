"""How fragile are the allocation decisions? Margins to the top-k boundary against seed noise (THESIS_GUIDE.md Step 9.1).

A model retrained with another seed scores the same stock on the same day differently. Scores are compared
as daily cross-sectional z-scores, the scale of portfolio.topk's margin_z, so that seeds and models with
different score scales are comparable:

- seed_noise: per stock-day, the standard deviation of the z-scored score across the seeds of one model
  (two or three seeds), i.e. how much training luck moves the stock's position in the cross-section;
- fragility: for the stocks a portfolio holds, the ratio margin_z / seed noise of the same stock-day. A held
  stock with a ratio below 1 is closer to the boundary than the typical change a new seed makes to its
  score: whether it is held is decided by training luck rather than by the data. The share of such
  holdings is compared with the turnover actually observed between the portfolios of two seeds.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from xaifin.config import RESULTS_ROOT


def zscores(model: str, universe: str, seq_len: int, pred_len: int, seeds=None, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Test predictions of every seed and test year of a model, as daily z-scores: date, ticker, seed, z."""
    frames = []
    for path in sorted(Path(results_root, "Regression", model, universe, f"sl{seq_len}_pl{pred_len}").glob("seed*/y*/predictions.csv.gz")):
        seed = int(path.parent.parent.name[4:])
        if seeds is None or seed in seeds:
            frames.append(pd.read_csv(path, usecols=["date", "ticker", "score"], parse_dates=["date"]).assign(seed=seed))
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "seed", "z"])
    p = pd.concat(frames, ignore_index=True)
    g = p.groupby(["seed", "date"])["score"]
    p["z"] = (p["score"] - g.transform("mean")) / g.transform("std")
    return p[["date", "ticker", "seed", "z"]]


def seed_noise(z: pd.DataFrame) -> pd.DataFrame:
    """Per stock-day: the standard deviation of z across seeds and the number of seeds; days with one seed are dropped."""
    noise = z.groupby(["date", "ticker"])["z"].agg(noise="std", seeds="size").reset_index()
    return noise[noise["seeds"] >= 2]


def fragility(holdings: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """The held rows of a portfolio (portfolio.topk.topk_holdings) with the seed noise of their stock-day and ratio = margin_z / noise."""
    held = holdings[holdings["held"]].merge(noise, left_on=["prediction_date", "ticker"], right_on=["date", "ticker"], how="inner")
    held["ratio"] = held["margin_z"] / held["noise"]
    return held.drop(columns="date")


def summary(held: pd.DataFrame) -> dict:
    """Median seed noise and margin, and the share of holdings closer to the boundary than 1x and 0.5x their seed noise."""
    return {"holdings": len(held), "median_noise": held["noise"].median(), "median_margin": held["margin_z"].median(),
            "median_ratio": held["ratio"].median(), "share_below_noise": float((held["ratio"] < 1).mean()),
            "share_below_half_noise": float((held["ratio"] < 0.5).mean())}
