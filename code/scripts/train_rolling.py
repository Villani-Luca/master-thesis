"""Train models on the rolling windows and save them to the results store (THESIS_GUIDE.md Step 3).

    python code/scripts/train_rolling.py --models MASTER --universes dji --years 2020 --seeds 42
    python code/scripts/train_rolling.py --models all --universes dji nasdaq100 sx5e --years 2020-2024

Runs whose metrics.json exists are skipped (resumable); --force retrains them. Every message is
also appended to code/results/Regression/train_rolling.log.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA

import argparse
import warnings
from datetime import datetime

import torch

from xaifin.config import CLEAN_DATA, DEFAULT_SL_PL, RESULTS_ROOT, SEEDS
from xaifin.models.registry import ADAPTERS
from xaifin.training.rolling import run_grid


def years(text: str) -> list[int]:
    if "-" in text:
        first, last = map(int, text.split("-"))
        return list(range(first, last + 1))
    return [int(text)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["all"], help=f"'all' or some of {', '.join(ADAPTERS)}")
    parser.add_argument("--universes", nargs="+", default=["dji"])
    parser.add_argument("--years", nargs="+", default=["2020"], help="test years, e.g. 2020 2021 or 2020-2024")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--sl", type=int, default=DEFAULT_SL_PL[0], help="lookback T")
    parser.add_argument("--pl", type=int, default=DEFAULT_SL_PL[1], help="horizon L")
    parser.add_argument("--no-clean", action="store_true", help="keep the samples the data-quality filter discards")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--force", action="store_true", help="retrain runs that are done already")
    args = parser.parse_args()

    models = list(ADAPTERS) if args.models == ["all"] else args.models
    test_years = [y for text in args.years for y in years(text)]
    log_file = RESULTS_ROOT / "Regression" / "train_rolling.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}\n")

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.use_deterministic_algorithms(True, warn_only=True)
    log(f"train_rolling: models={models} universes={args.universes} years={test_years} seeds={args.seeds} "
        f"T={args.sl} L={args.pl} clean={CLEAN_DATA and not args.no_clean}")
    run_grid(models, args.universes, test_years, args.seeds, args.sl, args.pl, clean=CLEAN_DATA and not args.no_clean,
             device=args.device, force=args.force, log=log)


if __name__ == "__main__":
    main()
