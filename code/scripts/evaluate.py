"""Are the explanations trustworthy on every model, universe and horizon? (THESIS_GUIDE.md Step 6)

    python code/scripts/evaluate.py                                   # core universes, every model and horizon
    python code/scripts/evaluate.py --universes sp500 sxxp            # the big universes (on the H200)
    python code/scripts/evaluate.py --models MASTER --universes dji --horizons 5

Notebook 06 checks the methods on one run per model (DJI, L = 5, seed 42, 2020). This script repeats the
checks on one run per model x universe x horizon: seed 42, test year 2022 (in every universe, and a
volatile one), on N_DAYS rebalance dates spread over the year:
- sanity check of IG (xai.evaluation.sanity_check): per stock-day, the Spearman correlation between the
  attributions of the trained model and of the same model with its initial weights, per feature and per
  input; with the relative change of the weights from their initial values and the correlation of the
  trained and untrained scores;
- deletion curves of IG and Saliency on the score (xai.evaluation.deletion_curve), one stock at a time
  ('own', faithful for every model in notebook 06) and all stocks at once ('joint'), against a random order.
Outputs in results/xai/evaluation/<MODEL>/<universe>/sl<T>_pl<L>/seed42/y2022/: sanity.parquet,
deletion.parquet and summary.json; a run with its summary.json is skipped. Messages go to
results/xai/explain.log too.
"""

import argparse
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from xaifin.config import CORE_UNIVERSES, RESULTS_ROOT, TOP_K
from xaifin.models.registry import ADAPTERS, find_runs, load_run
from xaifin.xai.base import explain_dates
from xaifin.xai.evaluation import deletion_curve, sanity_check, untrained_copy
from xaifin.xai.gradient import IntegratedGradients, Saliency

SEED, YEAR, N_DAYS = 42, 2022, 5


def evaluate(adapter, k: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sanity check and deletion curves of one run, on N_DAYS rebalance dates spread over its test year."""
    dates = explain_dates(adapter)
    days = [dates[i] for i in np.unique(np.linspace(0, len(dates) - 1, N_DAYS).round().astype(int))]
    ig = IntegratedGradients()
    sanity = sanity_check(adapter, ig, days, "pred", k)
    untrained = untrained_copy(adapter)
    w1 = torch.cat([p.detach().flatten() for p in adapter.model.parameters()])
    w0 = torch.cat([p.detach().flatten() for p in untrained.model.parameters()])
    curves, score_rho, wanted = [], [], set(days)
    for batch in adapter.day_batches("test"):
        if batch.date not in wanted:
            continue
        score_rho.append(spearmanr(adapter.predict(batch), untrained.predict(batch))[0])
        for name, explainer in (("IG", ig), ("Saliency", Saliency())):
            attr = explainer.explain(adapter, batch, "pred", k)
            for mode in ("own", "joint"):
                curves.append(deletion_curve(adapter, batch, attr, "pred", k, mode=mode).assign(method=name, mode=mode))
    curves = pd.concat(curves, ignore_index=True)
    aopc = curves.groupby(["method", "mode", "order"])["drop"].mean()  # area over the perturbation curve
    summary = {"days": [str(pd.Timestamp(d).date()) for d in days], "k": k,
               "weight_change": float((w1 - w0).norm() / w0.norm()),
               "scores_trained_vs_untrained": float(np.nanmean(score_rho)),
               "sanity_feature_median": float(sanity["feature"].median()),
               "sanity_input_median": float(sanity["input"].median()),
               "aopc": {f"{m} {mode} {order}": float(v) for (m, mode, order), v in aopc.items()}}
    return sanity, curves, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(ADAPTERS))
    parser.add_argument("--universes", nargs="+", default=CORE_UNIVERSES)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 1, 20], help="horizons L")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args()

    log_file = RESULTS_ROOT / "xai" / "explain.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  evaluate: {message}\n")

    warnings.filterwarnings("ignore")
    runs = find_runs()
    runs = runs[runs["checkpoint"] & runs["model"].isin(args.models) & runs["universe"].isin(args.universes)
                & runs["pred_len"].isin(args.horizons) & (runs["seed"] == SEED) & (runs["test_year"] == YEAR)]
    runs = runs.assign(order=runs["pred_len"].map({h: i for i, h in enumerate(args.horizons)})).sort_values(
        ["order", "universe", "model"])
    log(f"{len(runs)} runs (seed {SEED}, test year {YEAR}, {N_DAYS} days each)")
    for i, run in enumerate(runs.itertuples(), 1):
        label = f"[{i}/{len(runs)}] {run.model} {run.universe} sl{run.seq_len}_pl{run.pred_len}"
        out = (RESULTS_ROOT / "xai" / "evaluation" / run.model / run.universe / f"sl{run.seq_len}_pl{run.pred_len}"
               / f"seed{SEED}" / f"y{YEAR}")
        if (out / "summary.json").exists():
            log(f"{label}: done already")
            continue
        t0 = time.time()
        sanity, curves, summary = evaluate(load_run(run.path, device=args.device), TOP_K[run.universe])
        out.mkdir(parents=True, exist_ok=True)
        sanity.to_parquet(out / "sanity.parquet")
        curves.to_parquet(out / "deletion.parquet")
        (out / "summary.json").write_text(json.dumps({**summary, "seconds": round(time.time() - t0, 1)}, indent=2))
        a = summary["aopc"]
        log(f"{label}: sanity {summary['sanity_feature_median']:.2f}, AOPC own IG {a['IG own attribution']:.3f} "
            f"vs random {a['IG own random']:.3f}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
