"""Per-universe data summaries for Step 1 (notebook 01_universes_and_data, app page 1_Data).

The feature files are large (0.7-26 GB each), so every file is read once in a streaming pass and
the results are cached as small parquet files under code/results/data_summary/<universe>/.
`summarize(universe)` returns them, computing and caching them on the first call.

All statistics cover the study period (config.STUDY_START to STUDY_END, 2015-2024), except
membership and coverage, which show the full history for context.
"""

import difflib
import re
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from xaifin.config import (
    ALPHA360_SEQ_LEN,
    BENCHMARK_INDEX,
    DEFAULT_SL_PL,
    MODEL_GROUPS,
    NATION,
    RESULTS_ROOT,
    ROLLING_WINDOWS,
    STUDY_END,
    STUDY_START,
    TEST_YEARS,
    UNIVERSE_NAMES,
)
from xaifin.data.features import FAMILIES, alpha158_names, alpha360_names, catalog, market_names
from xaifin.data.loading import alpha_path, load_constituents, load_info, load_market, load_prices
from xaifin.data.quality import discarded_samples, drop_discarded

SUMMARY_ROOT = RESULTS_ROOT / "data_summary"
TABLES = [
    "meta",
    "membership",
    "coverage",
    "label_stats",
    "label_stats_clean",
    "extreme_labels",
    "discarded",
    "missing",
    "alpha158_corr",
    "alpha158_clusters",
    "market_volatility",
]
REDUNDANT_ABS_CORR = 0.9  # |Spearman| at or above which two features count as near-duplicates
SAMPLE_WEEKDAY = 2  # Alpha158 correlations use every Wednesday of the study period
TRADING_DAYS = 252
# Tables that need the streaming scans of the feature files; the others take seconds to recompute.
SCAN_TABLES = ["meta", "missing", "alpha158_corr", "alpha158_clusters"]


def membership(universe: str) -> pd.DataFrame:
    """Number of index constituents at each month end (a missing start or end date is open-ended)."""
    c = load_constituents(universe)
    months = pd.date_range("2000-01-31", pd.Timestamp.today(), freq="ME")
    start = c["start"].fillna(pd.Timestamp.min).to_numpy()
    end = c["end"].fillna(pd.Timestamp.max).to_numpy()
    members = [int(((start <= d) & (end >= d)).sum()) for d in months.to_numpy()]
    return pd.DataFrame({"date": months, "members": members})


def coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """Number of tickers with a price on each trading day."""
    return prices.groupby("date").size().rename("tickers").reset_index()


def universe_by_test_year(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Index members (last month end before) and tickers with prices at the start of each test year.

    Tickers are the most seen on one of the first five trading days: exchanges close on different
    holidays (the LSE on 2022-01-03 and 2023-01-02), so the first day alone undercounts STOXX 600.
    """
    membership, coverage_ = tables["membership"], tables["coverage"]
    rows = []
    for year, window in ROLLING_WINDOWS.items():
        start = pd.Timestamp(window.start_test_date)
        members = membership.loc[membership["date"] < start, "members"]
        tickers = coverage_.loc[coverage_["date"] >= start, "tickers"]
        rows.append({
            "test_year": year,
            "index_members": int(members.iloc[-1]) if len(members) else 0,
            "tickers_with_prices": int(tickers.head(5).max()) if len(tickers) else 0,
        })
    return pd.DataFrame(rows)


def forward_returns(prices: pd.DataFrame, pred_len: int) -> pd.DataFrame:
    """FinBench label: (adj_close[t + L] - adj_close[t]) / adj_close[t], per instrument."""
    p = prices.sort_values(["instrument", "date"])
    future = p.groupby("instrument")["adj_close"].shift(-pred_len)
    return p.assign(label=(future - p["adj_close"]) / p["adj_close"])


def label_stats(prices: pd.DataFrame, pred_len: int, discarded: pd.DataFrame | None = None) -> pd.DataFrame:
    """Distribution of the forward return (before FinBench's cross-sectional z-score) per year,
    without the `discarded` samples if given (xaifin.data.quality)."""
    df = forward_returns(prices, pred_len)
    if discarded is not None:
        df = drop_discarded(df, discarded)
    df = df[(df["date"] >= STUDY_START) & (df["date"] <= STUDY_END)].dropna(subset=["label"])
    year = df["date"].dt.year.rename("year")
    stats = df.groupby(year)["label"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    stats = stats.rename(columns={
        "count": "n", "1%": "p01", "5%": "p05", "25%": "q1", "50%": "median", "75%": "q3", "95%": "p95", "99%": "p99",
    })
    stats["share_positive"] = (df["label"] > 0).groupby(year).mean()
    stats["pred_len"] = pred_len
    return stats.reset_index()


def test_years_as_member(constituents: pd.DataFrame, ticker: str) -> list[int]:
    """Test years whose universe contains `ticker`: index member on the first test day."""
    rows = constituents[constituents["ticker"] == ticker]
    years = []
    for year in TEST_YEARS:
        day = pd.Timestamp(ROLLING_WINDOWS[year].start_test_date)
        if ((rows["start"].isna() | (rows["start"] <= day)) & (rows["end"].isna() | (rows["end"] >= day))).any():
            years.append(year)
    return years


def extreme_labels(universe: str, prices: pd.DataFrame, pred_len: int, discarded: pd.DataFrame) -> pd.DataFrame:
    """Instruments whose price doubles or halves within the horizon somewhere in the study period.

    Some are real moves (a crash, a takeover), many are data errors: a bad price, a ticker mapped
    to the wrong company, a delisting recorded as a price of 0. `min_price` helps tell them apart.
    `test_years` lists the test universes containing the instrument; `affected_years` those whose
    rolling window (first training day to last test day) also contains one of these labels, the
    only cases that reach a model. `discarded_days` counts the labels the data-quality filter removes.
    """
    df = forward_returns(prices, pred_len)
    df = df[(df["date"] >= STUDY_START) & (df["date"] <= STUDY_END)]
    extreme = df[(df["label"] >= 1.0) | (df["label"] <= -0.5)]
    kept = drop_discarded(extreme, discarded)
    extreme = extreme.assign(discarded=~extreme.index.isin(kept.index))
    table = extreme.groupby("instrument").agg(
        days=("label", "size"), first=("date", "min"), last=("date", "max"),
        min_label=("label", "min"), max_label=("label", "max"), discarded_days=("discarded", "sum"),
    )
    table["min_price"] = df[df["instrument"].isin(table.index)].groupby("instrument")["adj_close"].min()
    constituents = load_constituents(universe)
    table["name"] = table.index.map(constituents.drop_duplicates("ticker").set_index("ticker")["name"])
    member_years = {ticker: test_years_as_member(constituents, ticker) for ticker in table.index}
    table["test_years"] = [", ".join(map(str, member_years[ticker])) for ticker in table.index]
    affected = []
    for ticker, dates in extreme.groupby("instrument")["date"]:
        windows = [ROLLING_WINDOWS[year] for year in member_years[ticker]]
        years = [y for y, w in zip(member_years[ticker], windows) if dates.between(w.start_date, w.end_date).any()]
        affected.append(", ".join(map(str, years)))
    table["affected_years"] = affected
    return table.reset_index().sort_values("max_label", ascending=False, ignore_index=True)


_NAME_NOISE = {
    "inc", "corp", "corporation", "company", "co", "the", "plc", "ltd", "limited", "holdings", "group", "sa",
    "se", "ag", "nv", "spa", "asa", "ab", "oyj", "class", "and", "of", "de", "cl", "com", "shares",
}


def _name_similarity(a: str, b: str) -> float:
    """Similarity of two company names: shared words (legal suffixes ignored) or character overlap."""
    words_a, words_b = ({w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _NAME_NOISE and len(w) > 1}
                        for name in (a, b))
    shared = len(words_a & words_b) / min(len(words_a), len(words_b)) if words_a and words_b else 0.0
    return max(shared, difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio())


def name_mismatches(universe: str, min_similarity: float = 0.5) -> pd.DataFrame:
    """Constituents whose name in the membership file differs from the company of the listing the
    prices come from (<universe>_info.csv). Most are renames or abbreviations (HP Inc, SSE PLC); some
    are tickers mapped to the wrong company, whose prices and features then belong to another firm.
    `test_years` as in `extreme_labels`."""
    constituents = load_constituents(universe)
    info = load_info(universe)[["instrument", "Name"]].rename(columns={"instrument": "ticker", "Name": "listing_name"})
    table = constituents.drop_duplicates("ticker").merge(info, on="ticker").dropna(subset=["name", "listing_name"])
    table["similarity"] = [_name_similarity(a, b) for a, b in zip(table["name"], table["listing_name"])]
    table = table[table["similarity"] < min_similarity].copy()
    table["test_years"] = [", ".join(map(str, test_years_as_member(constituents, t))) for t in table["ticker"]]
    return table[["ticker", "name", "listing_name", "similarity", "test_years"]].sort_values("similarity", ignore_index=True)


def discarded_by_ticker(universe: str, discarded: pd.DataFrame) -> pd.DataFrame:
    """The discarded samples of the study period per instrument and rule, largest first."""
    d = discarded[(discarded["date"] >= STUDY_START) & (discarded["date"] <= STUDY_END)]
    table = d.groupby(["instrument", "reason"]).agg(samples=("date", "size"), first=("date", "min"), last=("date", "max"))
    names = load_constituents(universe).drop_duplicates("ticker").set_index("ticker")["name"]
    table = table.reset_index()
    table.insert(1, "name", table["instrument"].map(names))
    return table.sort_values("samples", ascending=False, ignore_index=True)


def _scan(universe: str, kind: str, names: list[str], sample_weekday: int | None = None):
    """One streaming pass over a feature file, restricted to the study period.

    Returns the missing-value share per feature, the number of rows scanned and, if
    `sample_weekday` is given, all rows of that weekday (0 = Monday) as a sample.
    """
    types = {"instrument": pa.string(), "date": pa.string()} | {name: pa.float64() for name in names}
    reader = pacsv.open_csv(
        alpha_path(universe, kind),
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types=types),
    )
    missing = pd.Series(0, index=names, dtype="int64")
    rows, sample = 0, []
    for batch in reader:
        dates = batch.column("date")
        keep = pc.and_(pc.greater_equal(dates, STUDY_START), pc.less_equal(dates, STUDY_END))
        df = batch.filter(keep).to_pandas()
        if df.empty:
            continue
        rows += len(df)
        missing += df[names].isna().sum()
        if sample_weekday is not None:
            sample.append(df[pd.to_datetime(df["date"]).dt.dayofweek == sample_weekday])

    share = (missing / max(rows, 1)).rename("missing_share").rename_axis("feature").reset_index()
    return share.assign(group=kind), rows, (pd.concat(sample, ignore_index=True) if sample else None)


def feature_correlation(sample: pd.DataFrame) -> pd.DataFrame:
    """|Spearman| correlation between the Alpha158 features, pooled over the sampled days."""
    names = alpha158_names()
    ranks = sample[names].dropna().rank().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(ranks, rowvar=False))
    corr = np.nan_to_num(corr)  # a constant column has no correlation
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=names, columns=names)


def correlation_clusters(corr: pd.DataFrame, threshold: float = REDUNDANT_ABS_CORR) -> pd.DataFrame:
    """Groups of near-duplicate features: complete linkage on 1 - |rho|, cut so that every pair
    inside a cluster has |rho| >= threshold. `order` is the dendrogram order, for heatmaps."""
    dist = 1.0 - corr.to_numpy()
    dist = np.clip((dist + dist.T) / 2, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    tree = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    order = np.empty(len(labels), dtype=int)
    order[leaves_list(tree)] = np.arange(len(labels))
    df = pd.DataFrame({"feature": corr.index, "cluster": labels, "order": order})
    df["cluster_size"] = df.groupby("cluster")["feature"].transform("size")
    return df.sort_values("order").reset_index(drop=True)


def redundant_groups(clusters: pd.DataFrame) -> pd.DataFrame:
    """Clusters with more than one feature, largest first: size and member features."""
    multi = clusters[clusters["cluster_size"] > 1]
    groups = multi.groupby("cluster")["feature"].agg(size="size", features=", ".join)
    return groups.sort_values("size", ascending=False).reset_index(drop=True)


def redundant_pairs(corr: pd.DataFrame, threshold: float = REDUNDANT_ABS_CORR) -> pd.DataFrame:
    """Feature pairs with |rho| >= threshold, strongest first."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack(future_stack=True).dropna().rename("abs_spearman")
    pairs = pairs.rename_axis(["feature_a", "feature_b"]).reset_index()
    return pairs[pairs["abs_spearman"] >= threshold].sort_values("abs_spearman", ascending=False, ignore_index=True)


def market_volatility(universe: str, window: int = 20) -> pd.DataFrame:
    """Annualized realized volatility of each market index: the std of daily index returns over
    `window` days (std_price_change_<window>_<index> in the market file) x sqrt(252)."""
    market = load_market(universe)
    prefix = f"std_price_change_{window}_"
    cols = [c for c in market.columns if c.startswith(prefix)]
    vol = market.melt(id_vars="date", value_vars=cols, var_name="index", value_name="volatility").dropna()
    vol["index"] = vol["index"].str.removeprefix(prefix)
    vol["volatility"] *= np.sqrt(TRADING_DAYS)
    vol["date"] = pd.to_datetime(vol["date"])
    return vol.reset_index(drop=True)


def volatility_by_year(volatility: pd.DataFrame) -> pd.DataFrame:
    """Mean annualized volatility per year (rows) and index (columns), over the study period."""
    v = volatility[(volatility["date"] >= STUDY_START) & (volatility["date"] <= STUDY_END)]
    return v.pivot_table(index=v["date"].dt.year.rename("year"), columns="index", values="volatility", aggfunc="mean")


def _span(values: pd.Series) -> str:
    low, high = int(values.min()), int(values.max())
    return str(low) if low == high else f"{low}-{high}"


def universe_table(summaries: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """One row per universe (thesis table of universes): the test years with a non-empty universe,
    its size, the stock-days of the study period (all tickers in the feature files), the share of
    them discarded by the data-quality filter, the label noise (5-day return IQR after the filter,
    mean over years) and the volatility of the universe's own index."""
    rows = []
    for universe, s in summaries.items():
        by_year = universe_by_test_year(s)
        tested = by_year[by_year["index_members"] > 0]
        labels = s["label_stats_clean"]
        volatility = volatility_by_year(s["market_volatility"])[BENCHMARK_INDEX[universe]]
        rows.append({
            "universe": UNIVERSE_NAMES[universe],
            "region": NATION[universe].upper(),
            "test_years": _span(tested["test_year"]),
            "index_members": _span(tested["index_members"]),
            "tickers_with_prices": _span(tested["tickers_with_prices"]),
            "stock_days": int(s["meta"]["rows_alpha158"].iloc[0]),
            "discarded": s["discarded"]["samples"].sum() / s["meta"]["rows_alpha158"].iloc[0],
            "label_iqr": (labels["q3"] - labels["q1"]).mean(),
            "share_positive": (labels["share_positive"] * labels["n"]).sum() / labels["n"].sum(),
            "index_volatility": volatility.mean(),
            "most_volatile_year": int(volatility.idxmax()),
        })
    return pd.DataFrame(rows)


def model_groups_table() -> pd.DataFrame:
    """The two comparable model groups (thesis table of groups): models, input per stock and features."""
    cat = catalog()
    universe_of = {}
    for universe, nation in NATION.items():
        universe_of.setdefault(nation, universe)
    n_market = ", ".join(f"{len(market_names(u))} {nation.upper()}" for nation, u in universe_of.items())
    rows = []
    for group, models in MODEL_GROUPS.items():
        features = cat[cat["group"] == group]
        seq_len = DEFAULT_SL_PL[0] if group == "alpha158" else ALPHA360_SEQ_LEN
        rows.append({
            "group": group,
            "models": ", ".join(models),
            "input_per_stock": f"[{seq_len}, {len(features)}]",
            "constant_features": int(features["constant"].sum()),
            "market_gate_features": f"{n_market} (MASTER, MATCC)" if group == "alpha158" else "none",
            "families": ", ".join(f for f in FAMILIES if f in set(features["family"])),
        })
    return pd.DataFrame(rows)


def summary_exists(universe: str) -> bool:
    """True when the slow part (the feature-file scans) is cached; the other tables take seconds."""
    return all((SUMMARY_ROOT / universe / f"{table}.parquet").exists() for table in SCAN_TABLES)


def summarize(
    universe: str, refresh: bool = False, seq_len: int = DEFAULT_SL_PL[0], pred_len: int = DEFAULT_SL_PL[1]
) -> dict[str, pd.DataFrame]:
    """All Step 1 summaries of a universe, read from the cache or computed (and cached) if missing.
    The data-quality filter uses the Alpha158 input (`seq_len` days), the longest the models read.

    Computing reads the price file and both feature files once: 10-30 seconds for the core
    universes, about two minutes for SP500/SXXP (Alpha360 files of ~25 GB).
    """
    folder = SUMMARY_ROOT / universe
    paths = {table: folder / f"{table}.parquet" for table in TABLES}
    cached = {} if refresh else {table: pd.read_parquet(path) for table, path in paths.items() if path.exists()}
    if len(cached) == len(TABLES):
        return {table: cached[table] for table in TABLES}

    # Tables added after a universe was cached are computed alone, without rescanning the feature files.
    started = time.time()
    prices = load_prices(universe)
    discarded = discarded_samples(prices, universe, pred_len, seq_len)
    tables = {
        "membership": membership(universe),
        "coverage": coverage(prices),
        "label_stats": label_stats(prices, pred_len),
        "label_stats_clean": label_stats(prices, pred_len, discarded),
        "extreme_labels": extreme_labels(universe, prices, pred_len, discarded),
        "discarded": discarded_by_ticker(universe, discarded),
        "market_volatility": market_volatility(universe),
    }
    if all(table in cached for table in SCAN_TABLES):
        tables = {table: df for table, df in tables.items() if table not in cached}
        for table, df in tables.items():
            df.to_parquet(paths[table])
        return {table: cached.get(table, tables.get(table)) for table in TABLES}

    missing158, rows158, sample = _scan(universe, "alpha158", alpha158_names(), sample_weekday=SAMPLE_WEEKDAY)
    missing360, rows360, _ = _scan(universe, "alpha360", alpha360_names())
    corr = feature_correlation(sample)
    tables["missing"] = pd.concat([missing158, missing360], ignore_index=True)
    tables["alpha158_corr"] = corr
    tables["alpha158_clusters"] = correlation_clusters(corr)
    tables["meta"] = pd.DataFrame([{
        "universe": universe,
        "study_start": STUDY_START,
        "study_end": STUDY_END,
        "pred_len": pred_len,
        "rows_alpha158": rows158,
        "rows_alpha360": rows360,
        "correlation_sample_rows": len(sample.dropna()),
        "seconds": round(time.time() - started),
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    }])

    folder.mkdir(parents=True, exist_ok=True)
    for table, df in tables.items():
        df.to_parquet(paths[table])
    return {table: tables[table] for table in TABLES}
