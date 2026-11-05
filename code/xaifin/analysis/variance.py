"""Where the disagreement between explanations comes from: seed, model, year or universe (THESIS_GUIDE.md Step 7).

The global importance of every explained run (results/xai/<method>/.../global_<target>.parquet,
xai.aggregate) is collected into one long table, indexed by the run's cell (model, universe,
seq_len, pred_len, seed, test_year). Two views of the same question:

- pair_agreement: the agreement (analysis.agreement) of every pair of runs that differ in exactly
  one factor: two seeds of one model (training noise), two models on the same seed and year
  (architecture), two years of one model (non-stationarity), two universes (universe effect).
  Models of different feature groups share no feature, so they are only paired at the levels with
  common names (family, source, horizon, lag).
- decomposition: an ANOVA-style split of the variance of each name's importance across runs into
  model, universe and year main effects (type II sums of squares), their interactions, and the seed
  (the spread of the three seeds within each model x universe x year cell), averaged over names
  weighted by their mean importance. Main effects are not exactly additive when cells are missing
  (no 2020 for EURO STOXX 50); interactions are then the remainder, floored at 0.
"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from xaifin.analysis.agreement import agreement
from xaifin.config import MODEL_GROUPS, RESULTS_ROOT

KEYS = ["model", "universe", "seq_len", "pred_len", "seed", "test_year"]
FACTORS = {"seed": "seed", "model": "model", "year": "test_year", "universe": "universe"}
GROUP = {m: g for g, models in MODEL_GROUPS.items() for m in models}


def load_importance(method: str, target: str = "pred", level: str = "family", value: str = "importance",
                    results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """One row per (run, name): the run's keys, its feature group, the name and its importance.

    `method` is a folder of results/xai/ (ig_joint, saliency, permutation). For permutation the
    level is always family and `value` is a column of its global.parquet (d_RankIC, topk_change, ...).
    """
    pattern = "global.parquet" if method == "permutation" else f"global_{target}.parquet"
    frames, missing = [], 0
    for path in sorted(Path(results_root, "xai", method).glob(f"*/*/*/*/*/{pattern}")):
        model, universe, slpl, seed, year = path.parts[-6:-1]
        seq_len, pred_len = (int(v) for v in slpl[2:].split("_pl"))
        g = pd.read_parquet(path)
        if method == "permutation":
            g = g.rename(columns={"family": "name"}).assign(level="family")
        if value not in g.columns:
            missing += 1
            continue
        g = g[g["level"] == level][["name", value]].rename(columns={value: "importance"})
        frames.append(g.assign(model=model, universe=universe, seq_len=seq_len, pred_len=pred_len, seed=int(seed[4:]),
                               test_year=int(year[1:]), group=GROUP.get(model)))
    if missing:
        warnings.warn(f"{missing} {method} global files have no column {value!r}: run scripts/explain.py --reaggregate")
    if not frames:
        return pd.DataFrame(columns=KEYS + ["group", "name", "importance"])
    return pd.concat(frames, ignore_index=True)[KEYS + ["group", "name", "importance"]]


def vectors(table: pd.DataFrame) -> dict[tuple, pd.Series]:
    """Run keys -> importance vector indexed by name."""
    return {key: g.set_index("name")["importance"] for key, g in table.groupby(KEYS)}


def pair_agreement(table: pd.DataFrame, min_names: int = 3, **kwargs) -> pd.DataFrame:
    """Agreement of every pair of runs that differ in exactly one factor; one row per pair, with `factor`."""
    vecs = vectors(table)
    rows = []
    for factor, column in FACTORS.items():
        others = [k for k in KEYS if k != column]
        index = {k: i for i, k in enumerate(KEYS)}
        cells = {}
        for key in vecs:
            cells.setdefault(tuple(key[index[o]] for o in others), []).append(key)
        for cell, keys in cells.items():
            for a, b in combinations(sorted(keys), 2):
                if factor == "model" and GROUP.get(a[0]) != GROUP.get(b[0]) and _feature_level(vecs[a], vecs[b]):
                    continue
                m = agreement(vecs[a], vecs[b], **kwargs)
                if m["names"] >= min_names:
                    rows.append({"factor": factor, **dict(zip(others, cell)), "a": a[index[column]], "b": b[index[column]],
                                 "cross_group": GROUP.get(a[0]) != GROUP.get(b[0]), **m})
    return pd.DataFrame(rows)


def _feature_level(a: pd.Series, b: pd.Series) -> bool:
    # the groups share a few feature names (OPEN0, HIGH0, LOW0), but most names at every other level
    return len(a.index.intersection(b.index)) < 0.5 * min(len(a), len(b))


def decomposition(table: pd.DataFrame, common_names: bool = True) -> pd.DataFrame:
    """Share of the variance of the importance explained by model, universe, year, interactions and seed.

    One row per name (with its mean importance) plus a weighted 'all' row. With `common_names`, only
    the names that every run has (e.g. the families shared by both feature groups) are used.
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    data = table.copy()
    if common_names:
        runs = data.groupby(KEYS).ngroups
        keep = data.groupby("name")["importance"].count() == runs
        data = data[data["name"].isin(keep[keep].index)]
    rows = []
    for name, g in data.groupby("name"):
        g = g.assign(v=g["importance"].astype(float))
        total = ((g["v"] - g["v"].mean()) ** 2).sum()
        if total <= 0 or len(g) < 4:
            continue
        cell = g.groupby(["model", "universe", "test_year"])["v"].transform("mean")
        seed_ss = ((g["v"] - cell) ** 2).sum()
        factors = [f for f in ("model", "universe", "test_year") if g[f].nunique() > 1]
        main = {}
        if factors:
            fit = smf.ols("v ~ " + " + ".join(f"C({f})" for f in factors), data=g).fit()
            if fit.df_resid < 1:
                continue  # too few runs for this name (while the grid is being explained)
            ss = anova_lm(fit, typ=2)["sum_sq"]
            main = {f: ss[f"C({f})"] for f in factors}
        interactions = max(total - seed_ss - sum(main.values()), 0.0)
        shares = {f: main.get(f, 0.0) / total for f in ("model", "universe", "test_year")}
        rows.append({"name": name, "mean_importance": g["v"].mean(), **shares, "interactions": interactions / total,
                     "seed": seed_ss / total, "runs": len(g)})
    parts = ["model", "universe", "year", "interactions", "seed"]
    out = pd.DataFrame(rows, columns=["name", "mean_importance", "model", "universe", "test_year", "interactions", "seed",
                                      "runs"]).rename(columns={"test_year": "year"})
    if out.empty:
        return out
    w = out["mean_importance"].abs()
    weighted = {p: float(np.average(out[p], weights=w)) for p in parts}
    out = pd.concat([out, pd.DataFrame([{"name": "all (weighted)", "mean_importance": np.nan, **weighted,
                                          "runs": int(out["runs"].max())}])], ignore_index=True)
    return out
