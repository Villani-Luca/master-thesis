"""Readers for the files under code/data/<universe>/."""

from pathlib import Path

import pandas as pd

from xaifin.config import DATA_ROOT, NATION


def prices_path(universe: str) -> Path:
    """Daily OHLCV (raw and adjusted) of every ticker: code/data/<universe>/<universe>.csv."""
    return DATA_ROOT / universe / f"{universe}.csv"


def alpha_path(universe: str, kind: str) -> Path:
    """Feature file of a universe; `kind` is 'alpha158' or 'alpha360'."""
    return DATA_ROOT / universe / f"{universe}_{kind}.csv"


def market_path(universe: str) -> Path:
    """Path of the market gate features used by MASTER and MATCC.

    The file lives in the universe folder: code/data/<universe>/<nation>_market.csv.
    FinBench's MATCC/train.py:232 reads it from <data_path>/<nation>_market.csv instead
    (issue 5 in docs/model_notes.md), which does not exist in this layout.
    """
    return DATA_ROOT / universe / f"{NATION[universe]}_market.csv"


def load_market(universe: str) -> pd.DataFrame:
    """Market gate features, one row per date: 63 columns for US universes, 42 for EU."""
    return pd.read_csv(market_path(universe))


def load_prices(universe: str, columns: tuple[str, ...] = ("instrument", "date", "adj_close")) -> pd.DataFrame:
    """Selected columns of the price file, with `date` parsed."""
    df = pd.read_csv(prices_path(universe), usecols=list(columns))
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_info(universe: str) -> pd.DataFrame:
    """Company name and GICS classification of every ticker in the price file (<universe>_info.csv)."""
    return pd.read_csv(DATA_ROOT / universe / f"{universe}_info.csv")


def load_constituents(universe: str) -> pd.DataFrame:
    """Index membership history: ticker (EODHD code), name, start, end (NaT = open-ended).

    The files differ per universe in columns and date format (M/D/YYYY or ISO); all have
    Name, StartDate, EndDate and EODHD.
    """
    raw = pd.read_csv(DATA_ROOT / "constituents" / "eodhd" / f"{universe}.csv")
    return pd.DataFrame({
        "ticker": raw["EODHD"],
        "name": raw["Name"],
        "start": pd.to_datetime(raw["StartDate"], format="mixed"),
        "end": pd.to_datetime(raw["EndDate"], format="mixed"),
    })


def load_alpha(universe: str, kind: str = "alpha158") -> pd.DataFrame:
    """Feature file of a universe: instrument, date (ISO string), then the feature columns."""
    return pd.read_csv(alpha_path(universe, kind))


def active_tickers(universe: str, on_date: str) -> list[str]:
    """Tickers that are index members on `on_date`.

    Same rule as FinBench's filter_constituents_by_date (Regression/MASTER/utils.py): membership
    started strictly before `on_date` and has not ended before it. A missing start or end counts
    as open-ended. FinBench applies it on the first test day, so the model is trained and tested
    on the members of that day (survivorship as in FinBench, kept for comparability).
    """
    members = load_constituents(universe)
    day = pd.Timestamp(on_date)
    start = members["start"].fillna(pd.Timestamp.min)
    end = members["end"].fillna(pd.Timestamp.max)
    return members.loc[(start < day) & (end >= day), "ticker"].tolist()


def add_labels(df: pd.DataFrame, universe: str, pred_len: int) -> pd.DataFrame:
    """`df` with a last column `Label`: the forward `pred_len`-day return on adjusted close.

    Label(t) = adj_close(t + pred_len) / adj_close(t) - 1, counted in rows of the ticker's own
    price history, as FinBench's extract_labels (Regression/MASTER/train.py).
    """
    close = pd.read_csv(prices_path(universe), usecols=["date", "instrument", "adj_close"])
    close = close.sort_values(["instrument", "date"])
    close["Label"] = close.groupby("instrument")["adj_close"].transform(lambda x: (x.shift(-pred_len) - x) / x)
    return df.merge(close[["date", "instrument", "Label"]], on=["date", "instrument"], how="left")


def select_valid_tickers(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Rows of the tickers that have at least one row in [start_date, end_date] (FinBench select_valid_ticker)."""
    in_window = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return df[df["instrument"].isin(in_window["instrument"].unique())]


def alpha158_frame(universe: str, window, pred_len: int, market: bool = True) -> pd.DataFrame:
    """The input table of the Alpha158 models for one rolling window, built as FinBench's train.py.

    Columns: instrument, date, the 157 Alpha158 features, the market gate features (if `market`,
    for MASTER and MATCC), Label. Steps: keep the members on the first test day, merge the market
    features by date, add the label, keep the tickers with data in the training period.
    `window` is a config.RollingWindow. Dates stay ISO strings, as in FinBench.
    """
    df = load_alpha(universe, "alpha158")
    df = df[df["instrument"].isin(active_tickers(universe, window.start_test_date))]
    if market:
        df = df.merge(load_market(universe), how="left", on="date")
    df = add_labels(df, universe, pred_len)
    return select_valid_tickers(df, window.start_date, window.end_train_date)
