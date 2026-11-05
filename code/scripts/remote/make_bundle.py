"""One zip with what a GPU server needs to train the grid: the code, the runs already done (so they
are skipped), the guide for Claude (START_HERE.md) and, with --data, the datasets. Unzipped, it is
a `master-thesis/` folder.

    python code/scripts/remote/make_bundle.py --data code/data --out ~/Downloads/xaifin_bundle.zip

Files come from git (tracked or new, never the ignored ones), so checkpoints, notebooks and the
thesis stay out. From the data, a feature CSV is left out when its Parquet copy is there
(scripts/convert_data.py), and so are the files training never reads.

To explain runs on the server instead (START_HERE_XAI.md), the bundle also needs the checkpoints of
those runs (--runs, globs under results/Regression/), the reference run of check_xai.py
(--xai-reference) and the XAI guide (--guide); the data can go in a separate zip (--no-code), so that it
is sent only if the server does not have it already:

    python code/scripts/remote/make_bundle.py --runs "*/sp500" "*/sxxp" "MASTER/dji/sl20_pl5/seed42/y2020" \\
        --xai-reference MASTER/dji/sl20_pl5/seed42/y2020 --guide code/scripts/remote/START_HERE_XAI.md \\
        --out ~/Downloads/xaifin_xai_bundle.zip
    python code/scripts/remote/make_bundle.py --no-code --data code/data --universes sp500 sxxp dji \\
        --out ~/Downloads/xaifin_xai_data.zip
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SKIP = ("code/notebooks/", "code/app/", "code/results/legacy/", "code/results/data_summary/", "code/results/xai/",
        "code/results/xai_sampled_dates/", "code/results/portfolio/", "latex-thesis/", ".vscode/")
UNUSED_DATA = ("_tech.csv", "_corr_matrix.npz", "_wikidata.json", "_wikidata_matrix.npz", "cnnpred_market.csv",
               "_inc_matrix_sect_ind.npz")
STORED = (".parquet", ".npz", ".zip", ".pth")  # compressed already: zipping them again costs time and saves nothing
RUN_FILES = ("config.json", "model.pth", "metrics.json", "train_history.csv")  # what explaining a run reads, and its record
XAI_METHODS = ("ig_joint", "saliency", "permutation", "intrinsic")  # explain.py's default methods


def wanted_data(path: Path) -> bool:
    if path.name.endswith(UNUSED_DATA) or path.suffix == ".tmp":
        return False
    return not (path.name.endswith(("_alpha158.csv", "_alpha360.csv")) and path.with_suffix(".parquet").exists())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="folder with the datasets (code/data layout); left out if omitted")
    parser.add_argument("--universes", nargs="+", default=None,
                        help="only these universes of --data (constituents/ always comes along)")
    parser.add_argument("--runs", nargs="+", default=[],
                        help="globs under code/results/Regression/ whose runs come with their checkpoints, e.g. '*/sp500'")
    parser.add_argument("--xai-reference", default=None,
                        help="run (e.g. MASTER/dji/sl20_pl5/seed42/y2020) whose explanations go to xai_reference/")
    parser.add_argument("--guide", default=str(REPO / "code/scripts/remote/START_HERE.md"),
                        help="the guide put at the root as START_HERE.md")
    parser.add_argument("--no-code", action="store_true", help="only the data (and nothing from git)")
    parser.add_argument("--out", default=str(Path.home() / "Downloads" / "xaifin_bundle.zip"))
    args = parser.parse_args()

    files = []
    if not args.no_code:
        listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO, capture_output=True,
                                text=True, check=True).stdout.splitlines()
        files = [f for f in listed if not f.startswith(SKIP) and (REPO / f).is_file()]
    regression = REPO / "code/results/Regression"
    run_dirs = sorted({p.parent for pattern in args.runs for p in regression.glob(f"{pattern}/**/model.pth")})
    run_files = [p for d in run_dirs for p in (d / name for name in RUN_FILES) if p.exists()]
    in_git = {REPO / f for f in files}
    run_files = [p for p in run_files if p not in in_git]  # config and metrics are often tracked: no duplicate entries
    reference = []
    if args.xai_reference:
        reference = [p for m in XAI_METHODS for p in (REPO / "code/results/xai" / m / args.xai_reference).glob("*")
                     if p.is_file()]
    data = []
    if args.data:
        root = Path(args.data)
        folders = [root / u for u in args.universes] + [root / "constituents"] if args.universes else [root]
        data = sorted(p for folder in folders for p in folder.rglob("*") if p.is_file() and wanted_data(p))

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in files:
            z.write(REPO / f, Path("master-thesis") / f)
        if not args.no_code:
            z.write(args.guide, "master-thesis/START_HERE.md")
        for p in run_files:
            z.write(p, Path("master-thesis") / p.relative_to(REPO),
                    compress_type=zipfile.ZIP_STORED if p.suffix in STORED else zipfile.ZIP_DEFLATED)
        for p in reference:
            z.write(p, Path("master-thesis/xai_reference") / p.relative_to(REPO / "code/results/xai"),
                    compress_type=zipfile.ZIP_STORED if p.suffix in STORED else zipfile.ZIP_DEFLATED)
        for i, p in enumerate(data, 1):
            mode = zipfile.ZIP_STORED if p.suffix in STORED else zipfile.ZIP_DEFLATED
            z.write(p, Path("master-thesis/code/data") / p.relative_to(args.data), compress_type=mode)
            print(f"  data {i}/{len(data)}: {p.relative_to(args.data)}", flush=True)
    print(f"{len(files)} code/result files, {len(run_dirs)} runs with checkpoint, {len(reference)} reference files, "
          f"{len(data)} data files -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    if not data and not args.no_code:
        print("no --data: the datasets must be copied into master-thesis/code/data/ on the server")


if __name__ == "__main__":
    main()
