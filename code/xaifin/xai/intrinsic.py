"""Intrinsic explanations: the weights a model computes for itself (THESIS_GUIDE.md Step 5.1).

Supporting evidence only (attention is not explanation, Jain and Wallace 2019). The models are not
changed: MASTER's forward pass is replayed with its own modules; for MATCC and FinFormer read-only
forward hooks keep the inputs of the attention modules and the weights are recomputed from them with
the modules' parameters. Every replay is checked against the model's own output (`max_diff`).
One function per model (WEIGHTS): master_weights, matcc_weights, finformer_weights; run_weights runs it
on the explained days. HIST's concept weights and FactorVAE's factor exposures are not extracted.

MASTER (Li et al. 2024), per day:
- gate [F_alpha]: the feature gate, computed from the market features of the last input day, the same
  for every stock: how the market reweights the 157 Alpha158 features (mean 1). This is how the market
  features act, which the gradient methods with the day-mean baseline do not see;
- temporal [N, T]: the final temporal attention, the weight of each day of the window in the stock's
  representation;
- intra [N, T]: the intra-stock attention (between the days of a stock's window), averaged over heads
  and over the query days: how much each day is attended to;
- inter [N, N]: the inter-stock attention, averaged over heads and the days of the window: row i is
  how much stock i attends to each other stock.
"""

import numpy as np
import torch

from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter


def _heads(module, q, k, v, scale):
    """Per-head softmax(q k^T / scale) and the concatenated outputs, as MASTER's attention modules."""
    dim = int(module.d_model / module.nhead)
    weights, outputs = [], []
    for i in range(module.nhead):
        sl = slice(i * dim, None if i == module.nhead - 1 else (i + 1) * dim)
        w = torch.softmax(torch.matmul(q[:, :, sl], k[:, :, sl].transpose(1, 2)) / scale, dim=-1)
        weights.append(w)
        outputs.append(torch.matmul(w, v[:, :, sl]))
    return torch.stack(weights), torch.concat(outputs, dim=-1)


@torch.no_grad()
def master_weights(adapter: ModelAdapter, batch: DayBatch) -> dict:
    """MASTER's intrinsic weights for one day; see the module docstring."""
    model = adapter.model.eval()
    x = batch.x.to(adapter.device)
    src = x[:, :, :model.gate_input_start_index]
    gate = model.feature_gate(x[:, -1, model.gate_input_start_index:model.gate_input_end_index])
    h = src * gate.unsqueeze(1)
    linear, pos, t_att, s_att, temporal, decoder = model.layers
    h = pos(linear(h))
    # intra-stock (TAttention): attention over the T days of each stock
    z = t_att.norm1(h)
    intra, out = _heads(t_att, t_att.qtrans(z), t_att.ktrans(z), t_att.vtrans(z), 1.0)
    z = t_att.norm2(z + out)
    h = z + t_att.ffn(z)
    # inter-stock (SAttention): attention over the N stocks at each day of the window
    z = s_att.norm1(h)
    q, k, v = (f(z).transpose(0, 1) for f in (s_att.qtrans, s_att.ktrans, s_att.vtrans))
    inter, out = _heads(s_att, q, k, v, s_att.temperature)
    z = s_att.norm2(z + out.transpose(0, 1))
    h = z + s_att.ffn(z)
    # final temporal attention
    e = temporal.trans(h)
    lam = torch.softmax(torch.matmul(e, e[:, -1, :].unsqueeze(-1)).squeeze(-1), dim=1)
    score = decoder(torch.matmul(lam.unsqueeze(1), h).squeeze(1)).squeeze(-1)
    max_diff = float((score - adapter.forward(x, batch.extras)).abs().max())
    return {"gate": gate[0].cpu().numpy(), "temporal": lam.cpu().numpy(), "intra": intra.mean((0, 2)).cpu().numpy(),
            "inter": inter.mean((0, 1)).cpu().numpy(), "score": score.cpu().numpy(), "max_diff": max_diff}


class _Capture:
    """Forward hooks that keep the inputs (and outputs) of some modules during one forward pass; read-only."""

    def __init__(self, modules: dict):
        self.inputs, self.outputs = {}, {}
        self.handles = [m.register_forward_hook(self._hook(name)) for name, m in modules.items()]

    def _hook(self, name):
        def hook(module, inputs, output):
            self.inputs[name], self.outputs[name] = inputs, output
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def matcc_weights(adapter: ModelAdapter, batch: DayBatch) -> dict:
    """MATCC (Cao et al. 2024), per day:
    - shift [F_alpha]: the market filter's output, added to the Alpha158 features (MATCC's counterpart of MASTER's
      multiplicative gate), the same for every stock;
    - inter [N, N]: inter-stock attention, averaged over heads and the days of the window;
    - temporal [N, T]: final temporal attention."""
    model = adapter.model.eval()
    x = batch.x.to(adapter.device)
    s_att, temporal = model.layers[3], model.layers[4]
    with _Capture({"s_att": s_att, "temporal": temporal}) as cap:
        score = adapter.forward(x, batch.extras)
    # MATCC calls feature_gate.forward() directly, which skips hooks: apply it to the same inputs
    shift = model.feature_gate(x[:, :, model.gate_input_start_index:model.gate_input_end_index])[:, 0, :]
    z = s_att.norm1(cap.inputs["s_att"][0])
    q, k, v = (f(z).transpose(0, 1) for f in (s_att.qtrans, s_att.ktrans, s_att.vtrans))
    inter, _ = _heads(s_att, q, k, v, s_att.temperature)
    e = temporal.trans(cap.inputs["temporal"][0])
    lam = torch.softmax(torch.matmul(e, e[:, -1, :].unsqueeze(-1)).squeeze(-1), dim=1)
    lam_check = float((torch.matmul(lam.unsqueeze(1), cap.inputs["temporal"][0]).squeeze(1) - cap.outputs["temporal"]).abs().max())
    return {"shift": shift[0].cpu().numpy(), "shift_spread": float((shift - shift[0]).abs().max()),
            "inter": inter.mean((0, 1)).cpu().numpy(), "temporal": lam.cpu().numpy(), "score": score.cpu().numpy(),
            "max_diff": lam_check}


@torch.no_grad()
def finformer_weights(adapter: ModelAdapter, batch: DayBatch) -> dict:
    """FinFormer (Zu et al. 2023), per day:
    - fusion [N]: mean of the gated-fusion gate z, the weight of the spatial (cross-stock) representation against
      the temporal one in each stock's final representation (1 = spatial only);
    - inter_dynamic [N, N]: the dynamic sparse attention among stocks, averaged over heads and "steps";
    - inter_static [N, N]: the attention on the static sector/industry graph, averaged over "steps";
    - steps [N, 60]: the attention of the spatial aggregation's last query over the 60 "steps". With FinBench's
      alphabetical column order (docs/model_notes.md issue 4) the steps are not days: steps 10s..10s+9 hold the 60
      lags of one series (CLOSE, HIGH, LOW, OPEN, VOLUME, VWAP), so this weighs the series."""
    model = adapter.model.eval()
    x, edge = adapter._inputs(batch.x, batch.extras)
    layer = model.ssdt.sparse_trans1
    with _Capture({"fusion": model.gate_fusion, "dyn": layer.dmhsa, "ssa": layer.ssa, "mha": model.s_aggr.mha}) as cap:
        score, _ = model(x, edge)
    hs, ht = cap.inputs["fusion"]
    z = torch.sigmoid(model.gate_fusion._fully_connected_xs(hs) + model.gate_fusion._fully_connected_xt(ht))
    dynamic = cap.outputs["dyn"][1]  # [steps, heads, N, N]
    static = cap.outputs["ssa"][1]  # [steps, N, N]
    steps = cap.outputs["mha"][1][:, -1, :]  # [N, steps]
    return {"fusion": z.mean(-1).cpu().numpy(), "inter_dynamic": dynamic.mean((0, 1)).cpu().numpy(),
            "inter_static": static.mean(0).cpu().numpy(), "steps": steps.cpu().numpy(), "score": score.cpu().numpy(),
            "max_diff": float((score - adapter.forward(batch.x, batch.extras)).abs().max())}


WEIGHTS = {"MASTER": master_weights, "MATCC": matcc_weights, "FinFormer": finformer_weights}


def run_weights(adapter: ModelAdapter, dates, weights_fn=None) -> dict:
    """The model's weights_fn (WEIGHTS) on every day of `dates`: one list per key, one entry per day."""
    weights_fn = weights_fn or WEIGHTS[adapter.cfg.model]
    wanted, out = set(dates), {"date": [], "tickers": [], "max_diff": 0.0}
    for batch in adapter.day_batches("test"):
        if batch.date in wanted:
            w = weights_fn(adapter, batch)
            out["date"].append(batch.date)
            out["tickers"].append(list(batch.tickers))
            out["max_diff"] = max(out["max_diff"], w.pop("max_diff"))
            w.pop("score")
            for key, value in w.items():
                out.setdefault(key, []).append(value)
    if "gate" in out:
        out["gate"] = np.stack(out["gate"])
    return out


def stack_weights(out: dict) -> dict:
    """run_weights' output as flat arrays for np.savez: per-day vectors stacked [D, ...]; per-stock vectors stacked over
    stock-days [rows, ...] with `row_date` and `ticker`; each N x N attention summarized per stock-day by its weight on
    the stock itself relative to uniform (`<key>_self`, 1 = uniform) and its normalized entropy (`<key>_entropy`,
    1 = uniform)."""
    flat = {"date": np.array(out["date"], dtype="datetime64[ns]")}
    flat["row_date"] = np.concatenate([[d] * len(t) for d, t in zip(out["date"], out["tickers"])]).astype("datetime64[ns]")
    flat["ticker"] = np.concatenate(out["tickers"])
    for key, values in out.items():
        if key in ("date", "tickers", "max_diff") or key.endswith("_spread"):
            continue
        first = np.asarray(values[0])
        if first.ndim == 2 and first.shape[0] == first.shape[1] and key.startswith("inter"):
            n = np.array([len(m) for m in values])
            flat[f"{key}_self"] = np.concatenate([np.diag(m) * len(m) for m in values])
            ent = [-(m * np.log(np.clip(m, 1e-12, None))).sum(1) / np.log(len(m)) for m in values]
            flat[f"{key}_entropy"] = np.concatenate(ent)
        elif first.ndim >= 1 and len(first) == len(out["tickers"][0]) and key != "gate" and key != "shift":
            flat[key] = np.concatenate([np.asarray(v) for v in values])
        else:
            flat[key] = np.stack([np.asarray(v) for v in values])
    return flat
