"""Parquet copies of the feature files (code/data/<universe>/<universe>_alpha158.csv and _alpha360.csv).

    python code/scripts/convert_data.py                     # every universe, skipping files already converted
    python code/scripts/convert_data.py --universes dji --check

The CSV is parsed by pandas in chunks, with the same settings as a plain pd.read_csv, so the Parquet
file holds exactly the values training would read from the CSV (--check compares the two in full;
only sensible on the small universes). xaifin.data.loading.load_alpha then reads the Parquet file,
and only the tickers it needs.
"""

import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xaifin.config import UNIVERSES
from xaifin.data.loading import alpha_path

CHUNK_ROWS = 200_000


def convert(csv, parquet) -> None:
    columns = pd.read_csv(csv, nrows=0).columns
    dtypes = {c: (str if c in ("instrument", "date") else "float64") for c in columns}
    schema = pa.schema([(c, pa.string() if c in ("instrument", "date") else pa.float64()) for c in columns])
    tmp = parquet.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for chunk in pd.read_csv(csv, dtype=dtypes, chunksize=CHUNK_ROWS):
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
    tmp.replace(parquet)


def check(csv, parquet) -> bool:
    a, b = pd.read_csv(csv), pd.read_parquet(parquet)
    return list(a.columns) == list(b.columns) and len(a) == len(b) and all(
        a[c].astype(str).equals(b[c].astype(str)) if c in ("instrument", "date")
        else a[c].to_numpy().tobytes() == b[c].to_numpy().tobytes() for c in a.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universes", nargs="+", default=UNIVERSES)
    parser.add_argument("--check", action="store_true", help="compare every converted file with its CSV")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for universe in args.universes:
        for kind in ("alpha158", "alpha360"):
            csv = alpha_path(universe, kind)
            parquet = csv.with_suffix(".parquet")
            if not csv.exists():
                print(f"{csv.name}: no CSV, {'Parquet present' if parquet.exists() else 'NO DATA'}", flush=True)
                continue
            if not parquet.exists() or args.force:
                print(f"{csv.name} ({csv.stat().st_size / 1e9:.1f} GB) -> {parquet.name}", flush=True)
                convert(csv, parquet)
            if args.check:
                print(f"  {parquet.name}: {'identical to the CSV' if check(csv, parquet) else 'DIFFERENT'}", flush=True)


if __name__ == "__main__":
    main()
