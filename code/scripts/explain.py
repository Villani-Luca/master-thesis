"""Attributions of the trained runs, saved to the results store (THESIS_GUIDE.md Step 5, PIANO.md B6).

    python code/scripts/explain.py --models MASTER --universes dji --years 2020
    python code/scripts/explain.py --methods ig saliency permutation --universes dji nasdaq100 sx5e
    python code/scripts/explain.py --horizons 1 20 --stride 4    # other horizons, every 4th rebalance date

For every run with a checkpoint (default: L = 5, the three core universes, every seed and test
year), on every other rebalance date of its test year (xai.base.explain_dates):
- ig, saliency: attributions_<target>.npz, params_<target>.json and global_<target>.parquet
  (xai.aggregate.global_importance) for both targets, pred and topk_margin;
- permutation: permutation.parquet (one row per day, family, repeat) and global.parquet (per family);
- intrinsic (MASTER, MATCC, FinFormer only): weights.npz with the models' own gates and attention
  (xai.intrinsic.stack_weights);
- shapley, lime (not run by default; PIANO.md B4: dji, test year 2020): values per stock and family,
  attributions_<target>.npz (attr [rows, families]) and global_<target>.parquet (level family).
Outputs that exist are skipped (resumable); nothing is overwritten, except with --reaggregate, which
rebuilds the global_<target>.parquet of ig and saliency from their saved attributions (after a change
of xai.aggregate) without recomputing them. Several processes can share the
GPU when they get different --models or --universes. Every message is also appended to
code/results/xai/explain.log.
"""

import argparse
import json
import time
import warnings
from datetime import datetime

import numpy as np

from xaifin.config import CORE_UNIVERSES, RESULTS_ROOT, SEEDS, TEST_YEARS, TOP_K
from xaifin.models.registry import ADAPTERS, find_runs, get_adapter, load_run, run_config
from xaifin.xai.aggregate import global_importance, input_table
from xaifin.xai.base import REBALANCE_STRIDE, TARGETS, explain_dates, explain_run, load_attributions, save_attributions, xai_dir
from xaifin.xai.gradient import IntegratedGradients, Saliency
from xaifin.xai.intrinsic import WEIGHTS, run_weights, stack_weights
from xaifin.xai.permutation import FamilyPermutation, summarize
from xaifin.xai.shap_lime import FamilyLIME, FamilyShapley, family_global

METHODS = {"ig": IntegratedGradients, "saliency": Saliency, "permutation": FamilyPermutation, "shapley": FamilyShapley,
           "lime": FamilyLIME, "intrinsic": None}
DEFAULT_METHODS = ["ig", "saliency", "permutation", "intrinsic"]
GROUP_METHODS = ("shapley", "lime")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(METHODS))
    parser.add_argument("--models", nargs="+", default=list(ADAPTERS))
    parser.add_argument("--universes", nargs="+", default=CORE_UNIVERSES)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5], help="horizons L")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--years", nargs="+", type=int, default=TEST_YEARS)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--stride", type=int, default=REBALANCE_STRIDE,
                        help="explain every n-th rebalance date (default: every other one)")
    parser.add_argument("--reaggregate", action="store_true", help="rebuild the global files of ig and saliency, then stop")
    args = parser.parse_args()

    log_file = RESULTS_ROOT / "xai" / "explain.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")

    warnings.filterwarnings("ignore")
    runs = find_runs()
    runs = runs[runs["checkpoint"] & runs["model"].isin(args.models) & runs["universe"].isin(args.universes)
                & runs["pred_len"].isin(args.horizons) & runs["seed"].isin(args.seeds) & runs["test_year"].isin(args.years)]
    if args.reaggregate:
        reaggregate(runs, args.targets, log)
        return
    log(f"explain: {len(runs)} runs, methods={args.methods}, targets={args.targets}")

    for i, run in enumerate(runs.itertuples(), 1):
        label = f"[{i}/{len(runs)}] {run.model} {run.universe} sl{run.seq_len}_pl{run.pred_len} seed{run.seed} y{run.test_year}"
        explainers = {m: METHODS[m]() for m in args.methods if METHODS[m] is not None}
        todo = [(m, t) for m in explainers if m != "permutation" for t in args.targets
                if not (xai_dir(explainers[m].name, _Cfg(run)) / f"global_{t}.parquet").exists()]
        todo_perm = "permutation" in args.methods and not (xai_dir("permutation", _Cfg(run)) / "global.parquet").exists()
        todo_intr = ("intrinsic" in args.methods and run.model in WEIGHTS
                     and not (xai_dir("intrinsic", _Cfg(run)) / "weights.npz").exists())
        if not todo and not todo_perm and not todo_intr:
            log(f"{label}: done already")
            continue
        t0 = time.time()
        adapter = load_run(run.path, device=args.device)
        k, dates = TOP_K[run.universe], explain_dates(adapter, args.stride)
        for method, target in todo:
            explainer = explainers[method]
            result = explain_run(adapter, explainer, target, k, dates, log=lambda m: None)
            if method in GROUP_METHODS:
                save_group_values(adapter, explainer, target, result, k)
                continue
            out = save_attributions(adapter, explainer.name, target, result, {**explainer.params(), "date_stride": args.stride})
            global_importance(result["attr"], result["date"], input_table(adapter)).to_parquet(out / f"global_{target}.parquet")
        if todo_perm:
            perm = explainers["permutation"]
            rows = perm.explain_run(adapter, dates, k)
            out = xai_dir(perm.name, adapter)
            out.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(out / "permutation.parquet")
            summarize(rows).to_parquet(out / "global.parquet")
            (out / "params.json").write_text(json.dumps({**perm.params(), "k": k, "days": len(dates), "date_stride": args.stride}, indent=2))
        if todo_intr:
            weights = run_weights(adapter, dates)
            out = xai_dir("intrinsic", adapter)
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "weights.npz", **stack_weights(weights))
            (out / "params.json").write_text(json.dumps({"method": "intrinsic", "model": run.model, "days": len(dates), "date_stride": args.stride,
                                                         "replay_max_diff": weights["max_diff"]}, indent=2))
        log(f"{label}: {', '.join(f'{m}/{t}' for m, t in todo)}{' permutation' if todo_perm else ''}"
            f"{' intrinsic' if todo_intr else ''}, "
            f"{len(dates)} days, {time.time() - t0:.0f} s")


def save_group_values(adapter, explainer, target: str, result: dict, k: int) -> None:
    """Shapley or LIME values [rows, families] and their family summary."""
    out = xai_dir(explainer.name, adapter)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"attributions_{target}.npz", attr=result["attr"].astype(np.float32), date=result["date"],
                        ticker=result["ticker"], score=result["score"].astype(np.float32), families=np.array(explainer.families))
    params = {**explainer.params(), "target": target, "k": k, "rows": int(len(result["ticker"])),
              "days": int(len(np.unique(result["date"]))), "seconds": round(result["seconds"], 1)}
    (out / f"params_{target}.json").write_text(json.dumps(params, indent=2))
    family_global(result["attr"], result["date"], explainer.families).to_parquet(out / f"global_{target}.parquet")


def reaggregate(runs, targets: list[str], log) -> None:
    """global_<target>.parquet of ig and saliency again from the saved attributions, for every explained run."""
    done = 0
    for run in runs.itertuples():
        cfg, _ = run_config(run.path)
        adapter = get_adapter(cfg.model)(cfg, device="cpu")  # only its feature names are needed
        inputs = input_table(adapter)
        for method in ("ig_joint", "saliency"):
            for target in targets:
                result = load_attributions(adapter, method, target)
                if result is not None:
                    global_importance(result["attr"], result["date"], inputs).to_parquet(
                        xai_dir(method, adapter) / f"global_{target}.parquet")
                    done += 1
    log(f"reaggregate: {done} global files rebuilt")


class _Cfg:
    """The fields of a find_runs row that xai_dir needs, without building the adapter."""

    def __init__(self, run):
        from xaifin.models.base import RunConfig

        self.cfg = RunConfig(run.model, run.universe, run.test_year, seed=run.seed, seq_len=run.seq_len, pred_len=run.pred_len)


if __name__ == "__main__":
    main()
