"""Check of the date sampling (RIESECUZIONE.md R4): global importance from all rebalancing dates (results/xai/) against
the first version on a subset of them (results/xai_sampled_dates/: one date in two at L = 5, one in four at L = 1 and 20).

For every run, method and target present in both: Spearman correlation of the feature importances, whether the family
ranking is identical, Spearman of the families, and the overlap of the 10 most important features. Prints the medians and
minima per method, target and horizon, and writes the per-run table to results/analysis/date_sampling.csv.

    python code/scripts/check_date_sampling.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from xaifin.config import RESULTS_ROOT

FULL, SAMPLED = RESULTS_ROOT / "xai", RESULTS_ROOT / "xai_sampled_dates"
CORE = ("dji", "nasdaq100", "sx5e")


def level(table: pd.DataFrame, name: str) -> pd.Series:
    return table[table["level"] == name].set_index("name")["importance"]


def compare(full: pd.DataFrame, sampled: pd.DataFrame) -> dict:
    out = {}
    f, s = level(full, "feature"), level(sampled, "feature")
    f, s = f.align(s, join="inner")
    if len(f):  # Shapley and LIME have families only
        out["feature_spearman"] = spearmanr(f, s).statistic
        out["top10_overlap"] = len(set(f.nlargest(10).index) & set(s.nlargest(10).index)) / 10
    f, s = level(full, "family"), level(sampled, "family")
    f, s = f.align(s, join="inner")
    out["family_spearman"] = spearmanr(f, s).statistic if len(f) > 2 else np.nan
    out["family_order_same"] = list(f.sort_values(ascending=False).index) == list(s.sort_values(ascending=False).index)
    return out


rows = []
for method in ("ig_joint", "saliency", "shapley", "lime"):
    for target in ("pred", "topk_margin"):
        for path in (SAMPLED / method).glob(f"*/*/sl*_pl*/seed*/y*/global_{target}.parquet"):
            model, universe, horizon, seed, year = path.parts[-6:-1]
            twin = FULL / path.relative_to(SAMPLED)
            if universe not in CORE or not twin.exists():
                continue
            rows.append({"method": method, "target": target, "model": model, "universe": universe,
                         "L": int(horizon.split("_pl")[1]), "seed": seed, "year": year,
                         **compare(pd.read_parquet(twin), pd.read_parquet(path))})

table = pd.DataFrame(rows)
out = RESULTS_ROOT / "analysis" / "date_sampling.csv"
out.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(out, index=False)
summary = table.groupby(["method", "target", "L"]).agg(
    runs=("model", "size"), feature_median=("feature_spearman", "median"), feature_min=("feature_spearman", "min"),
    family_median=("family_spearman", "median"), family_min=("family_spearman", "min"),
    family_order_same=("family_order_same", "mean"), top10_median=("top10_overlap", "median"))
pd.set_option("display.width", 200)
print(summary.round(3).to_string())
print(f"\n{len(table)} comparisons, written to {out}")
