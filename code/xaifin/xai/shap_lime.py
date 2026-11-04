"""Shapley values and LIME on feature families (THESIS_GUIDE.md Step 5.1, PIANO.md B4, decision D5).

The players are the feature families of the model's inputs: 7 for the Alpha158 models, 3 for the
Alpha360 models. A family is "absent" when its inputs are set, for every stock of the day,
to the day's average stock (the baseline of xai.gradient); the market family is identical across
stocks, so it is always at its baseline and is not a player. Masking the whole day at once keeps the
cross-sectional models' inputs consistent, the same choice as IG's joint path; every forward pass then
gives the value of the coalition for all stocks.

- shapley: exact Shapley values over the 2^P coalitions (128 forward passes a day for P = 7),
  phi_i(f) = sum_S |S|! (P - |S| - 1)! / P! [t_i(S + f) - t_i(S)], per stock and family;
- lime: LIME (Ribeiro et al. 2016) on the same binary representation: n_samples random coalitions,
  weighted by an exponential kernel of their distance from the full input, and a ridge regression per
  stock; its coefficients are the family attributions.
Both return [N, P] per day, families in the order of permutation.family_columns (attribute `families`).
"""

from itertools import product
from math import factorial

import numpy as np
import pandas as pd
import torch

from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter
from xaifin.xai.base import target_scores
from xaifin.xai.permutation import family_columns


def _masked_scores(adapter: ModelAdapter, batch: DayBatch, target: str, k: int, columns: list[np.ndarray],
                   coalitions: np.ndarray) -> np.ndarray:
    """t [len(coalitions), N]: the target with the families of each coalition present, the others at the day mean."""
    x = batch.x.to(adapter.device)
    base = x.mean(0, keepdim=True).expand_as(x)
    out = []
    with torch.no_grad():
        for coalition in coalitions:
            xm = x.clone()
            for present, cols in zip(coalition, columns):
                if not present:
                    xm[:, :, cols] = base[:, :, cols]
            out.append(target_scores(adapter.forward(xm, batch.extras), target, k).cpu().numpy())
    return np.stack(out)


class FamilyShapley:
    name = "shapley"

    def explain(self, adapter: ModelAdapter, batch: DayBatch, target: str, k: int) -> np.ndarray:
        cols = family_columns(adapter)
        self.families = list(cols)
        p = len(cols)
        coalitions = np.array(list(product([0, 1], repeat=p)), dtype=bool)
        values = _masked_scores(adapter, batch, target, k, [torch.as_tensor(c) for c in cols.values()], coalitions)
        index = {tuple(c): i for i, c in enumerate(coalitions)}
        phi = np.zeros((values.shape[1], p))
        for c, coalition in enumerate(coalitions):
            size = coalition.sum()
            for f in np.flatnonzero(~coalition):
                with_f = coalition.copy()
                with_f[f] = True
                weight = factorial(size) * factorial(p - size - 1) / factorial(p)
                phi[:, f] += weight * (values[index[tuple(with_f)]] - values[c])
        return phi

    def params(self) -> dict:
        return {"method": self.name, "players": "families", "absent": "day mean", "exact": True}


class FamilyLIME:
    name = "lime"

    def __init__(self, n_samples: int = 200, kernel_width: float = 0.75, ridge: float = 1.0, seed: int = 0):
        self.n_samples, self.kernel_width, self.ridge, self.seed = n_samples, kernel_width, ridge, seed

    def explain(self, adapter: ModelAdapter, batch: DayBatch, target: str, k: int) -> np.ndarray:
        cols = family_columns(adapter)
        self.families = list(cols)
        p = len(cols)
        rng = np.random.default_rng(self.seed)
        z = np.vstack([np.ones(p, dtype=bool), rng.random((self.n_samples - 1, p)) < 0.5])
        values = _masked_scores(adapter, batch, target, k, [torch.as_tensor(c) for c in cols.values()], z)
        distance = 1 - z.sum(1) / p  # share of families removed (cosine distance to the full input, as LIME's default for binary data)
        w = np.exp(-(distance**2) / self.kernel_width**2)
        design = np.hstack([np.ones((len(z), 1)), z.astype(float)])
        lhs = design.T @ (design * w[:, None]) + self.ridge * np.diag([0] + [1] * p)
        coef = np.linalg.solve(lhs, design.T @ (values * w[:, None]))  # [1 + P, N]
        return coef[1:].T

    def params(self) -> dict:
        return {"method": self.name, "players": "families", "absent": "day mean", "n_samples": self.n_samples,
                "kernel_width": self.kernel_width, "ridge": self.ridge, "seed": self.seed}


def family_global(values: np.ndarray, dates: np.ndarray, families: list[str]) -> pd.DataFrame:
    """Global importance of the families from per-stock values [rows, P], in the layout of
    aggregate.global_importance (level 'family'): share of |value| per sample, signed share, net (= importance
    here, one value per family), std over days."""
    total = np.abs(values).sum(1, keepdims=True)
    keep = total[:, 0] > 0
    share, signed = np.abs(values[keep]) / total[keep], values[keep] / total[keep]
    daily = pd.DataFrame(share, columns=families).groupby(pd.Series(dates[keep]).to_numpy()).mean()
    table = pd.DataFrame({"level": "family", "name": families, "importance": share.mean(0), "signed": signed.mean(0),
                          "net": share.mean(0), "std_days": daily.std().to_numpy(), "inputs": 1})
    table["rank"] = table["importance"].rank(ascending=False, method="min").astype(int)
    return table.sort_values("rank").reset_index(drop=True)
