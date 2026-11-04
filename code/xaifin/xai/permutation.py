"""Cross-sectional permutation importance of feature families (THESIS_GUIDE.md Step 5.1, decision D5).

On each explained day, the inputs of one family (all its columns, at every time step) are shuffled
across the stocks of the day with one permutation, so that each stock gets another stock's values
of that family and keeps its own values of the others. Then, against the unshuffled day:
- d_RankIC: RankIC of the scores with the labels the model is scored on (adapter.target), minus the
  same after shuffling; positive when the family carries ranking skill;
- topk_change: share of the top-k portfolio replaced (1 - |A & B| / k), the measure closest to the
  allocation;
- score_change: mean |change of score|, in standard deviations of the day's scores.
The market family is identical across the stocks of a day, so shuffling it changes nothing: it is
left out (its weight is read from the gradient methods instead).
"""

import numpy as np
import pandas as pd
import torch

from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter
from xaifin.training.metrics import daily_ic
from xaifin.xai.aggregate import feature_table


def family_columns(adapter: ModelAdapter) -> dict[str, np.ndarray]:
    """Family -> indices of its columns on the feature axis of x (the market family left out)."""
    families = feature_table(adapter)["family"]
    return {f: np.flatnonzero(families.to_numpy() == f) for f in families.unique() if f != "market"}


class FamilyPermutation:
    name = "permutation"

    def __init__(self, n_repeats: int = 5, seed: int = 0):
        self.n_repeats, self.seed = n_repeats, seed

    @torch.no_grad()
    def explain_day(self, adapter: ModelAdapter, batch: DayBatch, k: int, columns: dict[str, np.ndarray],
                    rng: np.random.Generator) -> list[dict]:
        x = batch.x.to(adapter.device)
        label = adapter.target(batch).cpu().numpy()
        base = adapter.forward(x, batch.extras).cpu().numpy()
        base_rankic = daily_ic(base, label)[1]
        base_top = set(np.argsort(base)[-k:])
        scale = base.std(ddof=1)
        rows = []
        for family, cols in columns.items():
            cols_t = torch.as_tensor(cols, device=x.device)
            for r in range(self.n_repeats):
                perm = torch.as_tensor(rng.permutation(len(x)), device=x.device)
                xp = x.clone()
                xp[:, :, cols_t] = x[perm][:, :, cols_t]
                s = adapter.forward(xp, batch.extras).cpu().numpy()
                rows.append({"date": batch.date, "family": family, "repeat": r, "base_RankIC": base_rankic,
                             "d_RankIC": base_rankic - daily_ic(s, label)[1],
                             "topk_change": 1 - len(base_top & set(np.argsort(s)[-k:])) / k,
                             "score_change": float(np.abs(s - base).mean() / scale)})
        return rows

    def explain_run(self, adapter: ModelAdapter, dates: list[pd.Timestamp], k: int) -> pd.DataFrame:
        """One row per (day, family, repeat)."""
        adapter.model.eval()
        columns, wanted, rng = family_columns(adapter), set(dates), np.random.default_rng(self.seed)
        rows = []
        for batch in adapter.day_batches("test"):
            if batch.date in wanted:
                rows += self.explain_day(adapter, batch, k, columns, rng)
        return pd.DataFrame(rows)

    def params(self) -> dict:
        return {"method": self.name, "n_repeats": self.n_repeats, "seed": self.seed, "unit": "family"}


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per family: mean of d_RankIC, topk_change and score_change over days and repeats, and the std over days."""
    daily = rows.groupby(["family", "date"])[["d_RankIC", "topk_change", "score_change"]].mean()
    table = daily.groupby("family").agg(["mean", "std"])
    table.columns = [f"{m}_{s}" if s == "std" else m for m, s in table.columns]
    return table.sort_values("topk_change", ascending=False).reset_index()
