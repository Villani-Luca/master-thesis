"""The results store on disk: shareable CSV copies of the outputs, and the checkpoint archive (THESIS_GUIDE.md §2.2).

What stays in git (small, readable without the code): per run config.json, metrics.json,
train_history.csv and predictions.csv.gz; per portfolio metrics.json, holdings.csv.gz and
returns.csv.gz. What does not (.gitignore): model.pth, the FinBench results pickle and the
parquet files; the checkpoints are shared as one zip (pack_checkpoints) instead.

- write_predictions_csv(run_dir): predictions.csv.gz from the run's results pickle (date, realized,
  ticker, score, label; topk.load_predictions).
- write_portfolio_csv(portfolio_dir): holdings.csv.gz and returns.csv.gz from the parquet files.
- export_csv(): both, for every run and portfolio of the store that lacks them.
- pack_checkpoints(zip_path): every model.pth with its config.json, under their results/ paths.
"""

import zipfile
from pathlib import Path

import pandas as pd

from xaifin.config import RESULTS_ROOT

PREDICTIONS_CSV = "predictions.csv.gz"
CSV_OPTIONS = {"index": False, "float_format": "%.6g", "date_format": "%Y-%m-%d"}


def write_predictions_csv(run_dir: Path) -> Path:
    """predictions.csv.gz of a run: one row per stock and last input day."""
    from xaifin.portfolio.topk import load_predictions

    out = Path(run_dir) / PREDICTIONS_CSV
    load_predictions([run_dir]).to_csv(out, **CSV_OPTIONS)
    return out


def write_portfolio_csv(portfolio_dir: Path) -> list[Path]:
    """holdings.csv.gz and returns.csv.gz of a portfolio, from its parquet files."""
    written = []
    for name in ("holdings", "returns"):
        source = Path(portfolio_dir) / f"{name}.parquet"
        if source.exists():
            out = source.with_name(f"{name}.csv.gz")
            pd.read_parquet(source).to_csv(out, **CSV_OPTIONS)
            written.append(out)
    return written


def export_csv(results_root: Path = RESULTS_ROOT, force: bool = False, log=print) -> None:
    """CSV copies for every run (with a results pickle) and portfolio of the store; existing ones are kept unless force."""
    root = Path(results_root)
    runs = sorted(p.parent for p in root.glob("Regression/*/*/*/*/*/results_sl*_pl*.pkl"))
    todo = [r for r in runs if force or not (r / PREDICTIONS_CSV).exists()]
    for run_dir in todo:
        write_predictions_csv(run_dir)
    log(f"predictions: {len(todo)} written, {len(runs) - len(todo)} already there")

    portfolios = sorted(p.parent for p in root.glob("portfolio/*/*/*/*/*/holdings.parquet"))
    todo = [p for p in portfolios if force or not (p / "holdings.csv.gz").exists()]
    for portfolio_dir in todo:
        write_portfolio_csv(portfolio_dir)
    log(f"portfolios: {len(todo)} written, {len(portfolios) - len(todo)} already there")


def pack_checkpoints(zip_path: Path, results_root: Path = RESULTS_ROOT, log=print) -> Path:
    """One zip with every model.pth and its config.json, at their paths under results/ (unzip into code/)."""
    root = Path(results_root)
    checkpoints = sorted(root.glob("Regression/*/*/*/*/*/model.pth"))
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for checkpoint in checkpoints:
            for f in (checkpoint, checkpoint.with_name("config.json")):
                if f.exists():
                    z.write(f, Path("results") / f.relative_to(root))
    log(f"{len(checkpoints)} checkpoints -> {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    return zip_path
