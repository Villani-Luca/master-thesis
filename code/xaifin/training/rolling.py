"""Rolling-window training: one run = one cell of the grid, saved to the results store (THESIS_GUIDE.md §2.2, Step 3).

A run trains one model on one universe, test year and seed with trainer.fit, evaluates the kept
weights on the test year and writes, in RunConfig.run_dir:

- model.pth, config.json (ModelAdapter.save, plus a 'training' section: selection rule, kept epoch,
  epochs run, validation metrics of the kept epoch, time, device, versions);
- metrics.json: test metrics of the kept weights (metrics.summary);
- train_history.csv: one row per epoch (trainer.fit);
- results_sl<T>_pl<L>.pkl: FinBench's results format (metrics, preds, labels, pred_date, last_date,
  tickers), so FinBench's Evaluation/ code can read it.

run_grid skips the runs whose metrics.json exists, so an interrupted grid resumes where it stopped.
"""

import json
import pickle
import platform
import time
from datetime import datetime
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
import torch

from xaifin.config import rolling_window
from xaifin.data.loading import active_tickers, trading_days
from xaifin.models.base import ModelAdapter, RunConfig
from xaifin.models.registry import find_runs, get_adapter
from xaifin.training.trainer import SELECTION_RULES, evaluate, fit


def _test_results(adapter: ModelAdapter) -> tuple[dict, dict]:
    """Test metrics of the current weights, and FinBench's results pickle."""
    preds, labels, metrics = evaluate(adapter, "test")
    dataset = adapter.splits()["test"]
    calendar = trading_days(adapter.cfg.universe)
    position = {d: i for i, d in enumerate(calendar)}
    last = [d.strftime("%Y-%m-%d") for d in dataset.dates]
    horizon = adapter.label_horizon
    realized = [calendar[position[d] + horizon] if position[d] + horizon < len(calendar) else None for d in last]
    results = {
        "metrics": metrics,
        "preds": [p[:, None] for p in preds],
        "labels": [y[:, None] for y in labels],
        "pred_date": realized,
        "last_date": last,
        "tickers": [list(t) for t in dataset.tickers],
    }
    return metrics, results


def is_done(cfg: RunConfig) -> bool:
    return (cfg.run_dir / "metrics.json").exists()


def has_test_universe(cfg: RunConfig) -> bool:
    """False when no stock is an index member at the start of the test year (EU universes in 2020)."""
    return bool(active_tickers(cfg.universe, rolling_window(cfg.test_year).start_test_date))


def train_run(cfg: RunConfig, device: str | None = None, log: Callable[[str], None] = print) -> dict:
    """Train one run, evaluate it on the test year and save everything to cfg.run_dir; returns its test metrics."""
    start = time.time()
    adapter = get_adapter(cfg.model)(cfg, device=device)
    log(f"{cfg.model} {cfg.universe} y{cfg.test_year} seed {cfg.seed} (T={cfg.seq_len}, L={cfg.pred_len}, "
        f"clean={cfg.clean}) on {adapter.device}: {SELECTION_RULES[adapter.hparams['selection']]}")
    days = {split: len(dataset) for split, dataset in adapter.splits().items()}
    log(f"  days: {days}")
    result = fit(adapter, log)
    metrics, results = _test_results(adapter)

    adapter.save()
    run_dir = cfg.run_dir
    history = result["history"]
    kept = history[history["epoch"] == result["kept_epoch"]].iloc[0]
    config = json.loads((run_dir / "config.json").read_text())
    config["training"] = {
        "selection": adapter.hparams["selection"],
        "selection_rule": SELECTION_RULES[adapter.hparams["selection"]],
        "kept_epoch": int(result["kept_epoch"]),
        "epochs_run": int(result["epochs_run"]),
        "valid_metrics_kept_epoch": {k[len("valid_"):]: float(v) for k, v in kept.items()
                                     if k.startswith("valid_") and not np.isnan(v)},
        "days": days,
        "label_horizon": adapter.label_horizon,
        "seconds": round(time.time() - start, 1),
        "device": str(adapter.device) + (f" ({torch.cuda.get_device_name(0)})" if adapter.device.type == "cuda" else ""),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    history.to_csv(run_dir / "train_history.csv", index=False)
    with open(run_dir / f"results_sl{cfg.seq_len}_pl{cfg.pred_len}.pkl", "wb") as f:
        pickle.dump(results, f)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))  # last: marks the run as done
    log(f"  kept epoch {result['kept_epoch']} of {result['epochs_run']}; test IC {metrics['IC']:+.4f} "
        f"RankIC {metrics['RankIC']:+.4f} MSE {metrics['MSE']:.4f} ({time.time() - start:.0f}s) -> {run_dir}")
    return metrics


def run_grid(models, universes, years, seeds, seq_len: int, pred_len: int, clean: bool, device: str | None = None,
             force: bool = False, log: Callable[[str], None] = print) -> None:
    """Train every (model, universe, year, seed) not done yet, in that order; errors are logged and skipped."""
    cells = list(product(models, universes, years, seeds))
    for k, (model, universe, year, seed) in enumerate(cells, start=1):
        cfg = RunConfig(model, universe, year, seed=seed, seq_len=seq_len, pred_len=pred_len, clean=clean)
        prefix = f"[{k}/{len(cells)}]"
        if is_done(cfg) and not force:
            log(f"{prefix} done already: {cfg.run_dir}")
            continue
        if not has_test_universe(cfg):
            log(f"{prefix} skipped: no {universe} index members at the start of {year}")
            continue
        log(prefix)
        try:
            train_run(cfg, device, log)
        except Exception as error:  # keep the grid going; the run stays undone and is retried next time
            log(f"{prefix} FAILED {model} {universe} y{year} seed {seed}: {type(error).__name__}: {error}")


def collect_results(results_root=None) -> pd.DataFrame:
    """One row per finished run: its cell, test metrics and how it was trained (config.json 'training').

    Columns: model, universe, seq_len, pred_len, seed, test_year, clean, the test metrics (IC, RankIC,
    ICIR, RankICIR, IC_pooled, MSE, MAE, RMSE, R2), kept_epoch, epochs_run, minutes, valid_IC (the
    kept epoch's validation IC), path.
    """
    runs = find_runs() if results_root is None else find_runs(results_root)
    rows = []
    for _, run in runs[runs["metrics"] & runs["config.json"]].iterrows():
        metrics = json.loads((run["path"] / "metrics.json").read_text())
        training = json.loads((run["path"] / "config.json").read_text()).get("training", {})
        rows.append({
            **run[["model", "universe", "seq_len", "pred_len", "seed", "test_year", "clean"]].to_dict(),
            **metrics,
            "kept_epoch": training.get("kept_epoch"), "epochs_run": training.get("epochs_run"),
            "minutes": round(training.get("seconds", np.nan) / 60, 1),
            "valid_IC": training.get("valid_metrics_kept_epoch", {}).get("IC"),
            "path": run["path"],
        })
    return pd.DataFrame(rows)
