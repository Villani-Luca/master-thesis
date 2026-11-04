"""Gradient attributions: Saliency and Integrated Gradients (THESIS_GUIDE.md Step 5.1).

Captum's attribution classes expand the inputs along dimension 0 to batch the integration steps,
which here would mix the stocks of a day (dimension 0 is the stock). The two methods are therefore
written on top of xaifin.xai.base.own_gradients, and IG uses Captum's integration nodes
(approximation_parameters, Gauss-Legendre by default), so it approximates the same integral.

IG of stock i, baseline 0:  attr_i = (x_i - 0) * integral_0^1 d t_i / d x_i (path(a)) da, along one of
- path "own" (the guide's definition, §2.3): only stock i's inputs move, x_i = a x_i, the other
  stocks stay at their values. Completeness holds per stock: sum(attr_i) = t_i(x) - t_i(x with x_i = 0).
  Costs n_steps x N forward and backward passes per day.
- path "joint": the whole day moves together, x = a x, and stock i keeps the part of the gradient
  that goes through its own inputs. n_steps forward passes per day; no per-stock completeness.
For a model that scores each stock on its own inputs only, the two are identical.

Baselines: "zero" (the training median of every feature) or "mean" (the day's cross-sectional mean
input [T, F], i.e. the average stock of the day; the path then runs from b to x and IG explains
t_i(x) - t_i(b)). The zero baseline does not suit FactorVAE: its first layer is a LayerNorm over the
features of each day of the window, so its score does not change when x_i is scaled (f(a x_i) = f(x_i)
for a > 0); along the path from 0 the score jumps at a = 0 and IG's per-feature values depend on
where the integration nodes fall.
"""

import numpy as np
import torch
from captum.attr._utils.approximation_methods import approximation_parameters

from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter
from xaifin.xai.base import own_gradients, target_scores


class Saliency:
    """The signed gradient d t_i / d x_i at the inputs (Simonyan et al. 2014; take |.| when aggregating)."""

    name = "saliency"

    def explain(self, adapter: ModelAdapter, batch: DayBatch, target: str, k: int) -> np.ndarray:
        return own_gradients(adapter, batch.x, batch.extras, target, k).cpu().numpy()

    def params(self) -> dict:
        return {"method": self.name}


class IntegratedGradients:
    """Integrated Gradients (Sundararajan et al. 2017) with baseline 0, per stock (see the module docstring)."""

    # Defaults decided 2026-11-04 (PIANO.md): 16 steps (converged on every model tested), joint path, day-mean baseline.
    def __init__(self, n_steps: int = 16, path: str = "joint", baseline: str = "mean", method: str = "gausslegendre"):
        if path not in ("own", "joint"):
            raise ValueError(f"path must be 'own' or 'joint', not {path!r}")
        if baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}, not {baseline!r}")
        self.n_steps, self.path, self.baseline, self.method = n_steps, path, baseline, method
        self.name = "ig" if path == "own" else "ig_joint"
        step_sizes, alphas = approximation_parameters(method)
        self.weights, self.alphas = step_sizes(n_steps), alphas(n_steps)

    def explain(self, adapter: ModelAdapter, batch: DayBatch, target: str, k: int) -> np.ndarray:
        x = batch.x.to(adapter.device)
        b = baseline_input(x, self.baseline)
        total = torch.zeros_like(x)
        if self.path == "joint":
            for w, a in zip(self.weights, self.alphas):
                total += w * own_gradients(adapter, b + a * (x - b), batch.extras, target, k)
        else:
            for i in range(len(x)):
                for w, a in zip(self.weights, self.alphas):
                    xi = x.clone()
                    xi[i] = b[i] + a * (x[i] - b[i])
                    xi.requires_grad_(True)
                    t = target_scores(adapter.forward(xi, batch.extras), target, k)[i]
                    (g,) = torch.autograd.grad(t, xi)
                    total[i] += w * g[i]
        return ((x - b) * total).cpu().numpy()

    def params(self) -> dict:
        return {"method": self.name, "n_steps": self.n_steps, "path": self.path, "baseline": self.baseline,
                "approximation": self.method}


BASELINES = ("zero", "mean")


def baseline_input(x: torch.Tensor, baseline: str) -> torch.Tensor:
    """The baseline of every stock, [N, T, F]: zeros, or the day's cross-sectional mean input."""
    if baseline == "zero":
        return torch.zeros_like(x)
    return x.mean(0, keepdim=True).expand_as(x)


def completeness_gap(adapter: ModelAdapter, batch: DayBatch, attr: np.ndarray, target: str, k: int,
                     baseline: str = "zero") -> np.ndarray:
    """Per stock: sum(attr_i) - (t_i(x) - t_i(x with x_i = b_i)). Near 0 for IG with path 'own'."""
    x = batch.x.to(adapter.device)
    b = baseline_input(x, baseline)
    with torch.no_grad():
        full = target_scores(adapter.forward(x, batch.extras), target, k)
        gap = np.empty(len(x))
        for i in range(len(x)):
            xi = x.clone()
            xi[i] = b[i]
            base = target_scores(adapter.forward(xi, batch.extras), target, k)[i]
            gap[i] = attr[i].sum() - (full[i] - base).item()
    return gap
