"""Readers for the files under code/data/<universe>/."""

from pathlib import Path

import pandas as pd

from xaifin.config import DATA_ROOT, NATION


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
