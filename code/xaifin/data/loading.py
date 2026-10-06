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
