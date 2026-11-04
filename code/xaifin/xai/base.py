"""What every explanation method shares: targets, dates, own-input gradients, storage (THESIS_GUIDE.md Step 5).

An explainer attributes, for one day, each stock's target to that stock's own inputs: a method
`explain(adapter, batch, target, k) -> attr[N, T, F]`. Five of the six models mix the stocks of a
day (attention, graphs, factors), so a stock's score also depends on the other stocks' inputs; as
the guide prescribes (§2.3), the attribution of stock i only covers x_i, with the other stocks'
inputs held at their values, and the whole day always goes through the model.

Targets (decision D4 of PIANO.md):
- "pred": the score s_i.
- "topk_margin": s_i minus the score stock i has to beat to be in the top k, i.e. the k-th highest
  score among the other stocks. Positive for the held stocks (then it is s_i - s_(k+1)), negative for
  the others (s_i - s_(k)), the signed version of portfolio.topk's margin. It explains membership of
  the portfolio: in a cross-sectional model, moving x_i also moves the other scores and the boundary.

Days: the prediction days the long-only portfolio trades on (portfolio.topk), on the L = 5 grid for
every horizon, so that attributions of different horizons are explained on the same days; every
other one of them (REBALANCE_STRIDE, decided 2026-11-04 to halve the cost), about 26 days a year.
Baselines are set by each method (xai.gradient: the day's mean input by default).

Storage, under results/xai/<method>/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<YEAR>/:
attributions.npz (attr float16 [rows, T, F] with the day and ticker of each row; days have different
N, so rows are stacked) and params.json.
"""

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from xaifin.config import RESULTS_ROOT, TOP_K
from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter
from xaifin.portfolio.topk import MAX_PREDICTION_AGE_DAYS, rebalance_periods

TARGETS = ("pred", "topk_margin")
REBALANCE_EVERY = 5  # L = 5 business days for every horizon (PIANO.md B2)
REBALANCE_STRIDE = 2  # explain every other rebalance date
GRAD_CHUNK_BYTES = 2**30  # memory for the batched own-input gradients (N x [N, T, F] floats)


class Explainer(Protocol):
    name: str

    def explain(self, adapter: ModelAdapter, batch: DayBatch, target: str, k: int) -> np.ndarray:
        """Attributions [N, T, F] of each stock's target to its own inputs, for one day."""

    def params(self) -> dict:
        """The settings saved in params.json."""


def target_scores(scores: torch.Tensor, target: str, k: int) -> torch.Tensor:
    """Per-stock target [N] from the day's scores [N]; differentiable (almost everywhere)."""
    if target == "pred":
        return scores
    if target != "topk_margin":
        raise ValueError(f"unknown target {target!r}; known: {TARGETS}")
    if len(scores) <= k:
        raise ValueError(f"topk_margin needs more than k = {k} stocks, the day has {len(scores)}")
    top = torch.topk(scores, k + 1).values  # top[k - 1]: k-th score, top[k]: (k+1)-th
    held = scores >= top[k - 1]
    return scores - torch.where(held, top[k], top[k - 1])


def own_gradients(adapter: ModelAdapter, x: torch.Tensor, extras: dict, target: str, k: int) -> torch.Tensor:
    """d t_i / d x_i for every stock i at the inputs x [N, T, F]: the diagonal blocks of the Jacobian.

    One forward pass; the backward passes are batched over stocks (is_grads_batched, in chunks that
    fit GRAD_CHUNK_BYTES), with a loop over stocks for the models whose operations cannot be batched.
    """
    x = x.to(adapter.device).detach().requires_grad_(True)
    t = target_scores(adapter.forward(x, extras), target, k)
    n = len(t)
    out = torch.empty_like(x)
    chunk = max(1, GRAD_CHUNK_BYTES // (x.numel() * x.element_size()))
    eye = torch.eye(n, device=t.device, dtype=t.dtype)
    try:
        for start in range(0, n, chunk):
            idx = torch.arange(start, min(start + chunk, n), device=t.device)
            (g,) = torch.autograd.grad(t, x, grad_outputs=eye[idx], retain_graph=True, is_grads_batched=True)
            out[idx] = g[torch.arange(len(idx)), idx]
    except RuntimeError:
        for i in range(n):
            (g,) = torch.autograd.grad(t[i], x, retain_graph=i < n - 1)
            out[i] = g[i]
    return out.detach()


def explain_dates(adapter: ModelAdapter, stride: int = REBALANCE_STRIDE) -> list[pd.Timestamp]:
    """The prediction days of the run's test year that the top-k portfolio trades on (portfolio.topk),
    every `stride`-th of them."""
    from xaifin.portfolio.backtest import expected_years

    days = np.array(sorted(b.date for b in adapter.day_batches("test")), dtype="datetime64[ns]")
    first = expected_years(adapter.cfg.universe)[0]
    year = adapter.cfg.test_year
    dates = set()
    for start, _ in rebalance_periods(f"{first}-01-01", f"{year + 1}-01-31", REBALANCE_EVERY):
        before = days[days < np.datetime64(start)]
        if len(before) and (start - pd.Timestamp(before[-1])).days <= MAX_PREDICTION_AGE_DAYS:
            dates.add(pd.Timestamp(before[-1]))
    return sorted(dates)[::stride]


def xai_dir(method: str, adapter: ModelAdapter) -> Path:
    c = adapter.cfg
    return (RESULTS_ROOT / "xai" / method / c.model / c.universe / f"sl{c.seq_len}_pl{c.pred_len}" / f"seed{c.seed}"
            / f"y{c.test_year}")


def explain_run(adapter: ModelAdapter, explainer: Explainer, target: str, k: int | None = None,
                dates: list[pd.Timestamp] | None = None, log=print) -> dict:
    """Attributions of every explained day of a run, stacked: attr [rows, T, F], date, ticker, score."""
    k = k or TOP_K[adapter.cfg.universe]
    wanted = set(dates or explain_dates(adapter))
    adapter.model.eval()
    attrs, day_col, tickers, scores = [], [], [], []
    t0 = time.time()
    for batch in adapter.day_batches("test"):
        if batch.date not in wanted:
            continue
        attrs.append(explainer.explain(adapter, batch, target, k))
        day_col += [batch.date] * len(batch.tickers)
        tickers += list(batch.tickers)
        scores.append(adapter.predict(batch))
    log(f"{explainer.name} {target}: {len(attrs)} days, {len(tickers)} rows, {time.time() - t0:.0f} s")
    return {"attr": np.concatenate(attrs), "date": np.array(day_col, dtype="datetime64[ns]"),
            "ticker": np.array(tickers), "score": np.concatenate(scores), "seconds": time.time() - t0}


def load_attributions(adapter: ModelAdapter, method: str, target: str) -> dict | None:
    """What save_attributions wrote (attr as float32, plus params), or None if the run is not explained yet."""
    out = xai_dir(method, adapter)
    path = out / f"attributions_{target}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        result = {key: z[key] for key in z.files}
    result["attr"] = result["attr"].astype(np.float32)
    result["params"] = json.loads((out / f"params_{target}.json").read_text())
    return result


def save_attributions(adapter: ModelAdapter, method: str, target: str, result: dict, params: dict) -> Path:
    """attributions_<target>.npz (float16) and params_<target>.json in xai_dir(method, adapter)."""
    out = xai_dir(method, adapter)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"attributions_{target}.npz", attr=result["attr"].astype(np.float16), date=result["date"],
                        ticker=result["ticker"], score=result["score"].astype(np.float32),
                        feature_names=np.array(adapter.feature_names))
    meta = {**params, "target": target, "k": TOP_K[adapter.cfg.universe], "date_stride": REBALANCE_STRIDE,
            "rows": int(len(result["ticker"])), "days": int(len(np.unique(result["date"]))),
            "seconds": round(result["seconds"], 1), "device": str(adapter.device)}
    (out / f"params_{target}.json").write_text(json.dumps(meta, indent=2))
    return out
