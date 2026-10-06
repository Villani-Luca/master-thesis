"""Feature names and their grouping into families, sources and horizons (THESIS_GUIDE.md Step 1).

Attributions are compared across models on these groups. Alpha158 and Alpha360 models share
almost no individual features, so they can only be compared at the level of family, source and
horizon. Each feature gets one value per dimension (columns of `catalog()`):

- family: what the feature measures (FAMILIES below).
- source: which raw series it is computed from (SOURCES below). The most robust level for
  Alpha158 vs Alpha360 comparisons, since it involves no interpretation.
- horizon: how far back it looks: short (last 5 days), medium (6-20), long (21-60). A rolling
  feature over w days gets the bucket of w; an Alpha360 value of lag k gets the bucket of k + 1,
  i.e. of the shortest Alpha158 window that contains that day.

Alpha360 values are ratios to today's close (volume: to today's volume), so CLOSE{k} is exactly
Alpha158's ROC{k}. Lagged price series are therefore mapped to momentum; at lag 0, OPEN/HIGH/LOW
and VWAP describe today's bar, like Alpha158's candle features. CLOSE0 (always 1) and VOLUME0
(~1) carry no information and are flagged as constant.

Names and orders match FinBench's generators (Evaluation/features/alpha158.py, alpha360.py) and
the columns of code/data/<universe>/<universe>_alpha{158,360}.csv. Market features differ between
US (3 indexes, 63 features) and EU (2 indexes, 42) and are read from the universe's market file.
"""

import re
from typing import NamedTuple

import pandas as pd

from xaifin.data.loading import market_path

FAMILIES = {
    "momentum": "direction and strength of recent price moves",
    "trend": "moving average and linear trend of the close",
    "position": "where the price sits within its recent range or distribution",
    "volatility": "dispersion of prices or returns",
    "candle": "shape of a single day's bar (open, high, low relative to close)",
    "volume": "level and changes of traded volume",
    "price_volume": "co-movement of price and volume",
    "market": "market-level gate features (MASTER, MATCC)",
}
SOURCES = {
    "close": "close price only",
    "ohlc": "open, high and/or low as well as close",
    "volume": "volume only",
    "price_volume": "price and volume together",
    "market": "market index price and volume",
}
HORIZONS = ["short", "medium", "long"]

WINDOWS = [5, 10, 20, 30, 60]
N_LAGS = 60


class Operator(NamedTuple):
    family: str
    source: str
    description: str


# Single-day Alpha158 features, in FinBench order.
ALPHA158_DAILY = {
    "KMID": Operator("candle", "ohlc", "(close - open) / open"),
    "KLEN": Operator("volatility", "ohlc", "(high - low) / open"),
    "KMID2": Operator("candle", "ohlc", "(close - open) / (high - low)"),
    "KUP": Operator("candle", "ohlc", "upper shadow / open"),
    "KUP2": Operator("candle", "ohlc", "upper shadow / (high - low)"),
    "KLOW": Operator("candle", "ohlc", "lower shadow / open"),
    "KLOW2": Operator("candle", "ohlc", "lower shadow / (high - low)"),
    "KSFT": Operator("candle", "ohlc", "(2 close - high - low) / open"),
    "KSFT2": Operator("candle", "ohlc", "(2 close - high - low) / (high - low)"),
    "OPEN0": Operator("candle", "ohlc", "open / close"),
    "HIGH0": Operator("candle", "ohlc", "high / close"),
    "LOW0": Operator("candle", "ohlc", "low / close"),
}

# Rolling Alpha158 operators, computed for every window in WINDOWS, in FinBench order.
ALPHA158_ROLLING = {
    "ROC": Operator("momentum", "close", "close w days ago / close"),
    "MA": Operator("trend", "close", "mean close / close"),
    "STD": Operator("volatility", "close", "std of close / close"),
    "BETA": Operator("trend", "close", "slope of the linear trend of close / close"),
    "RSQR": Operator("trend", "close", "R^2 of the linear trend of close"),
    "RESI": Operator("trend", "close", "residual of the linear trend of close / close"),
    "MAX": Operator("position", "ohlc", "max high / close"),
    "MIN": Operator("position", "ohlc", "min low / close"),
    "QTLU": Operator("position", "close", "80% quantile of close / close"),
    "QTLD": Operator("position", "close", "20% quantile of close / close"),
    "RANK": Operator("position", "close", "percentile of today's close in the window"),
    "RSV": Operator("position", "ohlc", "(close - min low) / (max high - min low)"),
    "IMAX": Operator("position", "ohlc", "position of the max high in the window / w"),
    "IMIN": Operator("position", "ohlc", "position of the min low in the window / w"),
    "IMXD": Operator("position", "ohlc", "IMAX - IMIN"),
    "CORR": Operator("price_volume", "price_volume", "corr(close, log volume)"),
    "CORD": Operator("price_volume", "price_volume", "corr(close change, log volume change)"),
    "CNTP": Operator("momentum", "close", "share of up days"),
    "CNTN": Operator("momentum", "close", "share of down days"),
    "CNTD": Operator("momentum", "close", "CNTP - CNTN"),
    "SUMP": Operator("momentum", "close", "gains / total absolute change (RSI-like)"),
    "SUMN": Operator("momentum", "close", "losses / total absolute change"),
    "SUMD": Operator("momentum", "close", "SUMP - SUMN"),
    "VMA": Operator("volume", "volume", "mean volume / volume"),
    "VSTD": Operator("volume", "volume", "std of volume / volume"),
    "WVMA": Operator("volatility", "price_volume", "coefficient of variation of |return| x volume"),
    "VSUMP": Operator("volume", "volume", "volume increases / total absolute volume change"),
    "VSUMN": Operator("volume", "volume", "volume decreases / total absolute volume change"),
    "VSUMD": Operator("volume", "volume", "VSUMP - VSUMN"),
}

# Alpha360 series, each for lags 0..N_LAGS-1, in FinBench order (lag-major: CLOSE0, OPEN0, ..., CLOSE1, ...).
ALPHA360_SERIES = {
    "CLOSE": Operator("momentum", "close", "close k days ago / close"),
    "OPEN": Operator("momentum", "ohlc", "open k days ago / close"),
    "HIGH": Operator("momentum", "ohlc", "high k days ago / close"),
    "LOW": Operator("momentum", "ohlc", "low k days ago / close"),
    "VOLUME": Operator("volume", "volume", "volume k days ago / volume"),
    "VWAP": Operator("momentum", "ohlc", "typical price (high + low + close) / 3 k days ago / close"),
}
_ALPHA360_CANDLE_AT_LAG0 = {"OPEN", "HIGH", "LOW", "VWAP"}
_ALPHA360_CONSTANT = {"CLOSE0", "VOLUME0"}

# Market statistics, one per index (and per window except price_change).
MARKET_STATS = {
    "price_change": "daily index return (adj close / previous - 1)",
    "mean_price_change": "mean daily index return over the window (market trend)",
    "std_price_change": "std of daily index returns over the window (market volatility)",
    "mean_vol": "mean index volume over the window / today's volume",
    "std_vol": "std of index volume over the window / today's volume",
}
_MARKET_NAME = re.compile(r"^((?:mean|std)_)?(price_change|vol)(?:_(\d+))?_(.+)$")


def horizon(window: int) -> str:
    """Horizon bucket of a look-back of `window` trading days."""
    if window <= 5:
        return "short"
    return "medium" if window <= 20 else "long"


def alpha158_names() -> list[str]:
    """The 157 Alpha158 feature names, in CSV / FinBench order."""
    return list(ALPHA158_DAILY) + [f"{op}{w}" for w in WINDOWS for op in ALPHA158_ROLLING]


def alpha360_names() -> list[str]:
    """The 360 Alpha360 column names, in CSV / FinBench order (lag-major)."""
    return [f"{series}{k}" for k in range(N_LAGS) for series in ALPHA360_SERIES]


def market_names(universe: str) -> list[str]:
    """Market gate feature names of a universe, in CSV order (63 for US, 42 for EU)."""
    return [c for c in pd.read_csv(market_path(universe), nrows=0).columns if c != "date"]


def _row(group, feature, operator, window, lag, op, horizon_days, constant=False, market_index=None):
    return {
        "group": group,
        "feature": feature,
        "operator": operator,
        "window": window,
        "lag": lag,
        "family": op.family,
        "source": op.source,
        "horizon": horizon(horizon_days),
        "constant": constant,
        "market_index": market_index,
        "description": op.description,
    }


def catalog(universe: str | None = None) -> pd.DataFrame:
    """One row per feature with its group, family, source and horizon.

    Columns: group ('alpha158' | 'alpha360' | 'market'), feature, operator, window (rolling
    window, or 1 for single-day features), lag (Alpha360 only), family, source, horizon,
    constant, market_index (market only), description. Market rows are included when `universe`
    is given, since they differ between US and EU. OPEN0, HIGH0 and LOW0 exist in both Alpha158
    and Alpha360 with the same definition, so the key is (group, feature).
    """
    rows = [_row("alpha158", name, name, 1, None, op, 1) for name, op in ALPHA158_DAILY.items()]
    rows += [
        _row("alpha158", f"{name}{w}", name, w, None, op, w)
        for w in WINDOWS
        for name, op in ALPHA158_ROLLING.items()
    ]

    for k in range(N_LAGS):
        for series, op in ALPHA360_SERIES.items():
            if k == 0 and series in _ALPHA360_CANDLE_AT_LAG0:
                op = op._replace(family="candle", description=op.description.replace(" k days ago", ""))
            feature = f"{series}{k}"
            rows.append(_row("alpha360", feature, series, None, k, op, k + 1, constant=feature in _ALPHA360_CONSTANT))

    if universe is not None:
        for name in market_names(universe):
            prefix, variable, window, index = _MARKET_NAME.match(name).groups()
            operator = f"{prefix or ''}{variable}"
            window = int(window) if window else 1
            op = Operator("market", "market", MARKET_STATS[operator])
            rows.append(_row("market", name, operator, window, None, op, window, market_index=index))

    df = pd.DataFrame(rows)
    df[["window", "lag"]] = df[["window", "lag"]].astype("Int64")
    return df
