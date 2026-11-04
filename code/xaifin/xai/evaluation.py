"""Are the explanations trustworthy? Sanity check and faithfulness (THESIS_GUIDE.md Step 6, PIANO.md C1).

- sanity_check (Adebayo et al. 2018): the attributions of the trained model against those of the same
  architecture with its initial weights (the adapter built for the run's seed and not trained, i.e. the
  weights training started from). If the two rank a stock's inputs alike, the method shows the inputs'
  structure rather than what the model learned.
- deletion_curve (Samek et al. 2017): for each stock, its inputs are set to the baseline (the day's average
  stock) in the order of their attribution, most positive first, a growing share at a time, all stocks of
  the day at once (mode 'joint', as IG's joint path) or one stock at a time with the others at their values
  (mode 'own', N forward passes per step); the target falls fast if the attribution is faithful. The same
  with a random order is the reference. AOPC = mean over the steps of t(x) - t(x with the top share removed), per stock; the
  RankIC of the masked scores is recorded too.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter
from xaifin.models.registry import get_adapter
from xaifin.training.metrics import daily_ic
from xaifin.xai.base import target_scores

FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def untrained_copy(adapter: ModelAdapter) -> ModelAdapter:
    """The run's architecture with the initial weights of its seed (never trained), sharing the loaded data."""
    copy = get_adapter(adapter.cfg.model)(adapter.cfg, device=adapter.device)
    copy._splits = adapter._splits
    copy.model.eval()
    return copy


def sanity_check(adapter: ModelAdapter, explainer, dates, target: str, k: int) -> pd.DataFrame:
    """Per stock-day: Spearman between the |attribution| per feature (summed over lags) of the trained and the
    untrained model, and the same per input."""
    untrained = untrained_copy(adapter)
    wanted, rows = set(dates), []
    for batch in adapter.day_batches("test"):
        if batch.date not in wanted:
            continue
        a = explainer.explain(adapter, batch, target, k)
        b = explainer.explain(untrained, batch, target, k)
        for i, ticker in enumerate(batch.tickers):
            rows.append({"date": batch.date, "ticker": ticker,
                         "feature": spearmanr(np.abs(a[i]).sum(0), np.abs(b[i]).sum(0))[0],
                         "input": spearmanr(np.abs(a[i]).ravel(), np.abs(b[i]).ravel())[0]})
    return pd.DataFrame(rows)


@torch.no_grad()
def deletion_curve(adapter: ModelAdapter, batch: DayBatch, attr: np.ndarray, target: str, k: int,
                   fractions=FRACTIONS, seed: int = 0, mode: str = "joint") -> pd.DataFrame:
    """One row per (order, fraction): mean drop of the target t(x) - t(masked) over stocks, and (mode 'joint') the
    RankIC of the masked scores with the day's labels. Orders: 'attribution' (most positive first) and 'random'."""
    x = batch.x.to(adapter.device)
    base = x.mean(0, keepdim=True).expand_as(x).reshape(len(x), -1)
    flat = x.reshape(len(x), -1)
    label = adapter.target(batch).cpu().numpy()
    full = target_scores(adapter.forward(x, batch.extras), target, k).cpu().numpy()
    rng = np.random.default_rng(seed)
    orders = {"attribution": np.argsort(-attr.reshape(len(x), -1), axis=1),
              "random": np.stack([rng.permutation(flat.shape[1]) for _ in range(len(x))])}
    rows = []
    for name, order in orders.items():
        order_t = torch.as_tensor(order, device=x.device)
        for q in fractions:
            n = max(1, int(round(q * flat.shape[1])))
            masked = flat.clone()
            idx = order_t[:, :n]
            masked.scatter_(1, idx, base.gather(1, idx))
            if mode == "own":
                t = np.empty(len(x))
                for i in range(len(x)):
                    xi = flat.clone()
                    xi[i] = masked[i]
                    t[i] = target_scores(adapter.forward(xi.reshape(x.shape), batch.extras), target, k)[i].item()
                rows.append({"date": batch.date, "order": name, "fraction": q, "drop": float(np.mean(full - t)),
                             "RankIC": np.nan})
                continue
            scores = adapter.forward(masked.reshape(x.shape), batch.extras)
            t = target_scores(scores, target, k).cpu().numpy()
            rows.append({"date": batch.date, "order": name, "fraction": q, "drop": float(np.mean(full - t)),
                         "RankIC": daily_ic(scores.cpu().numpy(), label)[1]})
    return pd.DataFrame(rows)


def aopc(curves: pd.DataFrame) -> pd.Series:
    """Area over the perturbation curve per order: the mean drop over fractions and days."""
    return curves.groupby("order")["drop"].mean()
