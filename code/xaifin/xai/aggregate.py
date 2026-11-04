"""From attributions [rows, T, F] to global importance per feature, lag, family, source, horizon (Step 5.2).

Each sample (one stock on one day) is normalized first, a~ = |a| / sum(|a|) over its T x F inputs,
so that models with different score scales and days with different volatility weigh the same. The
importance of a group is then the share of a~ it gets, averaged over samples; `signed` is the same
share with the sign of a (the direction of the effect), and `std_days` the standard deviation of the
daily averages (how much the importance moves from day to day). `net` sums the attributions of a
group before taking |.| (|sum a| over the group, as a share of the sum over groups): the group's total
contribution, which is what group-level methods (xai.shap_lime, permutation) measure, while
`importance` (sum |a|) also counts inputs of the group that push in opposite directions. On families
the two can rank differently stock by stock (MASTER, one run: Spearman with exact family Shapley 0.26
for importance, 0.87 for net) and agree globally (0.93 and 1.0).

The lag of an input: for the Alpha158 and market inputs, T - 1 - t (0 = the last input day); for the
Alpha360 models (T = 1), the lag of the column in the catalog. Families, sources and horizons come
from xaifin.data.features.catalog.
"""

import numpy as np
import pandas as pd

from xaifin.data.features import catalog, market_names
from xaifin.models.base import ModelAdapter

LEVELS = ("feature", "lag", "family", "source", "horizon")


def feature_table(adapter: ModelAdapter) -> pd.DataFrame:
    """One row per column of x (feature axis F), in order, with its catalog entry (family, source, horizon, lag)."""
    cat = catalog(adapter.cfg.universe).set_index(["group", "feature"])
    market = set(market_names(adapter.cfg.universe))
    keys = [("market" if f in market else adapter.group, f) for f in adapter.feature_names]
    table = cat.loc[keys].reset_index()
    table.index.name = "column"
    return table


def input_table(adapter: ModelAdapter) -> pd.DataFrame:
    """One row per input (t, column) of x [T, F], flattened in C order, with feature, family, source, horizon, lag."""
    features = feature_table(adapter)
    t_len = adapter.seq_len if adapter.group == "alpha158" else 1
    rows = features.loc[np.tile(features.index, t_len)].reset_index()
    rows.insert(0, "t", np.repeat(np.arange(t_len), len(features)))
    if adapter.group == "alpha158":
        rows["lag"] = t_len - 1 - rows["t"]
    rows["lag"] = rows["lag"].astype(int)
    return rows


def global_importance(attr: np.ndarray, dates: np.ndarray, inputs: pd.DataFrame) -> pd.DataFrame:
    """Importance of every feature, lag, family, source and horizon, from attributions [rows, T, F]."""
    flat = attr.reshape(len(attr), -1).astype(np.float64)
    total = np.abs(flat).sum(1, keepdims=True)
    keep = total[:, 0] > 0
    share, signed = np.abs(flat[keep]) / total[keep], flat[keep] / total[keep]
    day = pd.Series(dates[keep]).astype("datetime64[ns]").to_numpy()
    out = []
    for level in LEVELS:
        codes, names = pd.factorize(inputs[level].astype(str))
        group = np.zeros((len(names), flat.shape[1]))
        group[codes, np.arange(flat.shape[1])] = 1
        imp, sig = share @ group.T, signed @ group.T  # [samples, groups]
        net = np.abs(sig)
        net = net / np.where(net.sum(1, keepdims=True) > 0, net.sum(1, keepdims=True), 1)
        daily = pd.DataFrame(imp, columns=names).groupby(day).mean()
        table = pd.DataFrame({"level": level, "name": names, "importance": imp.mean(0), "signed": sig.mean(0),
                              "net": net.mean(0), "std_days": daily.std().to_numpy(), "inputs": group.sum(1).astype(int)})
        table["rank"] = table["importance"].rank(ascending=False, method="min").astype(int)
        out.append(table.sort_values("rank"))
    return pd.concat(out, ignore_index=True)
