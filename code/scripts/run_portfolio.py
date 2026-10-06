"""Long-only top-k portfolios from the trained runs' test predictions (THESIS_GUIDE.md Step 4).

    python code/scripts/run_portfolio.py                 # every complete (model, universe, seed)
    python code/scripts/run_portfolio.py --partial --force

Writes results/portfolio/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/top<k>/ (holdings.parquet,
returns.parquet, metrics.json). Portfolios that exist are skipped unless --force.
"""

import argparse
import warnings

from xaifin.portfolio.backtest import run_portfolio_grid
from xaifin.training.rolling import collect_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="default: every trained model")
    parser.add_argument("--universes", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=None, help="stocks held (default: config.TOP_K per universe)")
    parser.add_argument("--partial", action="store_true", help="also models whose test years are not all trained")
    parser.add_argument("--force", action="store_true", help="recompute existing portfolios")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    results = collect_results()
    if args.models:
        results = results[results["model"].isin(args.models)]
    if args.universes:
        results = results[results["universe"].isin(args.universes)]
    run_portfolio_grid(results, k=args.k, partial=args.partial, force=args.force)


if __name__ == "__main__":
    main()
