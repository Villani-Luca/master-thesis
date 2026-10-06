"""Model name -> adapter class, and trained runs -> ready adapters (THESIS_GUIDE.md §2.2-2.3).

Scripts, notebooks and the app pick a model by name (`get_adapter("HIST")`) and reopen a trained
run from its folder in the results store (`load_run`).
"""

import json
import re
from pathlib import Path

import pandas as pd
import torch

from xaifin.config import RESULTS_ROOT
from xaifin.models.base import ModelAdapter, RunConfig
from xaifin.models.discoverplf import DiscoverPLFAdapter
from xaifin.models.factorvae import FactorVAEAdapter
from xaifin.models.finformer import FinFormerAdapter
from xaifin.models.hist import HISTAdapter
from xaifin.models.master import MASTERAdapter
from xaifin.models.matcc import MATCCAdapter

ADAPTERS: dict[str, type[ModelAdapter]] = {
    a.name: a for a in (MASTERAdapter, MATCCAdapter, FactorVAEAdapter, HISTAdapter, DiscoverPLFAdapter, FinFormerAdapter)
}

# .../<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<YEAR>, under Regression/ or results/legacy/
_RUN_PATH = re.compile(r"(?:^|/)(?P<model>[^/]+)/(?P<universe>[^/]+)/sl(?P<seq_len>\d+)_pl(?P<pred_len>\d+)/"
                       r"seed(?P<seed>\d+)/y(?P<test_year>\d+)$")


def get_adapter(name: str) -> type[ModelAdapter]:
    """The adapter class of a model, by its name (case-insensitive): 'MASTER', 'HIST', ..."""
    for key, adapter in ADAPTERS.items():
        if key.lower() == name.lower():
            return adapter
    raise KeyError(f"unknown model {name!r}; known: {', '.join(ADAPTERS)}")


def run_config(run_dir: Path) -> tuple[RunConfig, bool]:
    """The RunConfig of a trained run, and whether it was read from config.json.

    Runs saved by ModelAdapter.save have a config.json. Older runs (the first MASTER runs of
    master_model.ipynb, now in results/legacy/) have none: their configuration is read from the
    folder names, with FinBench's hyper-parameters and clean=False, because they were trained
    before the data-quality filter existed.
    """
    run_dir = Path(run_dir)
    config_file = run_dir / "config.json"
    if config_file.exists():
        c = json.loads(config_file.read_text())
        fields = {k: c[k] for k in ("model", "universe", "test_year", "seed", "seq_len", "pred_len", "clean")}
        return RunConfig(**fields, hparams=c.get("hparams", {})), True
    match = _RUN_PATH.search(run_dir.resolve().as_posix())
    if match is None:
        raise ValueError(f"{run_dir} has no config.json and is not a results-store run folder")
    g = match.groupdict()
    return RunConfig(g["model"], g["universe"], int(g["test_year"]), seed=int(g["seed"]), seq_len=int(g["seq_len"]),
                     pred_len=int(g["pred_len"]), clean=False), False


def load_run(run_dir: Path, device: str | torch.device | None = None) -> ModelAdapter:
    """The adapter of a trained run with its weights loaded, in eval mode (data loaded on first use)."""
    cfg, _ = run_config(run_dir)
    adapter = get_adapter(cfg.model)(cfg, device=device)
    adapter.load(run_dir)
    adapter.model.eval()
    return adapter


def find_runs(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """One row per trained run (a folder with model.pth) of the results store, with what it holds."""
    rows = []
    for checkpoint in sorted(Path(results_root).glob("Regression/*/*/*/*/*/model.pth")):
        run_dir = checkpoint.parent
        try:
            cfg, has_config = run_config(run_dir)
        except ValueError:
            continue
        rows.append({
            "model": cfg.model, "universe": cfg.universe, "seq_len": cfg.seq_len, "pred_len": cfg.pred_len,
            "seed": cfg.seed, "test_year": cfg.test_year, "clean": cfg.clean, "config.json": has_config,
            "metrics": (run_dir / "metrics.json").exists(), "history": (run_dir / "train_history.csv").exists(),
            "path": run_dir,
        })
    columns = ["model", "universe", "seq_len", "pred_len", "seed", "test_year", "clean", "config.json", "metrics",
               "history", "path"]
    return pd.DataFrame(rows, columns=columns)
