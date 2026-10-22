"""One zip with what a GPU server needs to train the grid: the code, the runs already done (so they
are skipped), the guide for Claude (START_HERE.md) and, with --data, the datasets. Unzipped, it is
a `master-thesis/` folder.

    python code/scripts/remote/make_bundle.py --data code/data --out ~/Downloads/xaifin_bundle.zip

Files come from git (tracked or new, never the ignored ones), so checkpoints, notebooks and the
thesis stay out. From the data, a feature CSV is left out when its Parquet copy is there
(scripts/convert_data.py), and so are the files training never reads.
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SKIP = ("code/notebooks/", "code/app/", "code/results/legacy/", "code/results/data_summary/", "latex-thesis/",
        ".vscode/")
UNUSED_DATA = ("_tech.csv", "_corr_matrix.npz", "_wikidata.json", "_wikidata_matrix.npz", "cnnpred_market.csv",
               "_inc_matrix_sect_ind.npz")
STORED = (".parquet", ".npz", ".zip")  # compressed already: zipping them again costs time and saves nothing


def wanted_data(path: Path) -> bool:
    if path.name.endswith(UNUSED_DATA) or path.suffix == ".tmp":
        return False
    return not (path.name.endswith(("_alpha158.csv", "_alpha360.csv")) and path.with_suffix(".parquet").exists())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="folder with the datasets (code/data layout); left out if omitted")
    parser.add_argument("--out", default=str(Path.home() / "Downloads" / "xaifin_bundle.zip"))
    args = parser.parse_args()

    listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout.splitlines()
    files = [f for f in listed if not f.startswith(SKIP) and (REPO / f).is_file()]
    data = sorted(p for p in Path(args.data).rglob("*") if p.is_file() and wanted_data(p)) if args.data else []

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in files:
            z.write(REPO / f, Path("master-thesis") / f)
        z.write(REPO / "code/scripts/remote/START_HERE.md", "master-thesis/START_HERE.md")
        for i, p in enumerate(data, 1):
            mode = zipfile.ZIP_STORED if p.suffix in STORED else zipfile.ZIP_DEFLATED
            z.write(p, Path("master-thesis/code/data") / p.relative_to(args.data), compress_type=mode)
            print(f"  data {i}/{len(data)}: {p.relative_to(args.data)}", flush=True)
    print(f"{len(files)} code/result files, {len(data)} data files -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    if not data:
        print("no --data: the datasets must be copied into master-thesis/code/data/ on the server")


if __name__ == "__main__":
    main()
