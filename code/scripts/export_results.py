"""Shareable copies of the results store (xaifin.store): CSV files for git, and the checkpoints as one zip.

    python code/scripts/export_results.py                            # CSV for every run and portfolio lacking them
    python code/scripts/export_results.py --zip checkpoints.zip      # also zip every model.pth (+ config.json)

New runs and portfolios write their CSV files already (train_rolling.py, run_portfolio.py); this
script is for the ones saved before, or with --force to rewrite them all.
"""

import argparse

from xaifin.store import export_csv, pack_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", default=None, help="write the checkpoints archive to this path")
    parser.add_argument("--force", action="store_true", help="rewrite existing CSV files")
    args = parser.parse_args()

    export_csv(force=args.force)
    if args.zip:
        pack_checkpoints(args.zip)


if __name__ == "__main__":
    main()
