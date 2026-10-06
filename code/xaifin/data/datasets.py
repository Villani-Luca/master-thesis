"""Daily cross-sectional datasets: one sample = one trading day = one DayBatch (THESIS_GUIDE.md §2.3).

Source: code/finbench/Regression/MASTER/load_dataset.py (CSVDataset) and train.py.
FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.

The samples are the same as FinBench's CSVDataset, in the same order: for each day, every ticker
whose last `seq_len` rows up to that day end on that day, tickers in alphabetical order. Changes:

- The windows are built with array indexing instead of FinBench's scan of every ticker's history
  for every day (same result, checked in the Step 2 equivalence test), and gathered per day on
  access instead of being stored, so a dataset holds the table only once.
- The label is returned apart from the features (FinBench keeps it as the last feature column).
- With `discarded` (config.CLEAN_DATA), the samples of xaifin.data.quality.discarded_samples are
  dropped after building the windows, so a window never joins the days before and after a gap.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from xaifin.config import CLEAN_DATA, RollingWindow, rolling_window
from xaifin.data.loading import alpha158_frame, load_prices
from xaifin.data.normalization import RobustZScore
from xaifin.data.quality import discarded_samples

SPLITS = ("train", "valid", "test")


@dataclass
class DayBatch:
    date: pd.Timestamp  # prediction ("last input") date
    tickers: list[str]  # N stocks in the cross-section that day
    x: torch.Tensor  # [N, T, F] normalized features
    y: torch.Tensor  # [N] raw forward return over pred_len days (NaN filled with 0, as FinBench)
    extras: dict = field(default_factory=dict)  # fixed non-feature inputs: market_value, stock2concept, ...


class DailyDataset(Dataset):
    """The days of one split, as DayBatch, from a table with instrument, date, features and Label.

    `start_date`/`end_date` bound the split. For valid and test, the table is read from
    `seq_len + pred_len - 1` trading days earlier, so that the first prediction date has a full
    window (FinBench). In every split the last `pred_len` rows of each ticker are dropped: their
    label would read prices after `end_date`.
    """

    def __init__(self, frame: pd.DataFrame, seq_len: int, pred_len: int, start_date: str, end_date: str,
                 normalizer: RobustZScore, split: str = "train", discarded: pd.DataFrame | None = None):
        self.seq_len, self.pred_len, self.split = seq_len, pred_len, split
        calendar = np.sort(frame["date"].unique())
        if split in ("valid", "test"):
            start_date = calendar[np.argmax(calendar >= start_date) - seq_len - pred_len + 1]

        table = frame[(frame["date"] >= start_date) & (frame["date"] <= end_date)]
        table = table.sort_values(["instrument", "date"], ignore_index=True)
        table = table[table.groupby("instrument").cumcount(ascending=False) >= pred_len]
        table = normalizer.transform(table).fillna(0).reset_index(drop=True)

        self.feature_names = [c for c in table.columns if c not in ("instrument", "date", "Label")]
        self.values = torch.tensor(table[self.feature_names].to_numpy(np.float32))
        self.labels = torch.tensor(table["Label"].to_numpy(np.float32))
        instrument, date = table["instrument"].to_numpy(), table["date"].to_numpy()

        # Rows that end a full window: at least seq_len - 1 earlier rows of the same ticker.
        ends = np.flatnonzero(table.groupby("instrument").cumcount().to_numpy() >= seq_len - 1)
        if discarded is not None:
            samples = pd.MultiIndex.from_arrays([instrument[ends], pd.to_datetime(date[ends])])
            ends = ends[~samples.isin(pd.MultiIndex.from_frame(discarded[["instrument", "date"]]))]
        ends = ends[np.argsort(date[ends], kind="stable")]  # by day, tickers stay alphabetical

        days, starts = np.unique(date[ends], return_index=True)
        self.rows = np.split(ends, starts[1:])  # per day: the row of each ticker's last input day
        self.dates = [pd.Timestamp(d) for d in days]
        self.tickers = [instrument[r].tolist() for r in self.rows]

        # FinBench's results pickle: last input date and the date the label is realized.
        position = {d: i for i, d in enumerate(calendar)}
        self.input_dates = list(days)
        self.output_dates = [calendar[position[d] + pred_len] for d in days]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> DayBatch:
        rows = torch.from_numpy(self.rows[i])
        window = rows[:, None] + torch.arange(1 - self.seq_len, 1)  # [N, T] row indices
        return DayBatch(self.dates[i], self.tickers[i], self.values[window], self.labels[rows])


def alpha158_splits(universe: str, test_year: int, seq_len: int, pred_len: int, market: bool = True,
                    clean: bool = CLEAN_DATA) -> dict[str, DailyDataset]:
    """Train, valid and test datasets of the Alpha158 models for one rolling window.

    `market=True` appends the market gate features (MASTER, MATCC). `clean` drops the samples with
    unusable prices (xaifin.data.quality); False reproduces FinBench.
    """
    window: RollingWindow = rolling_window(test_year)
    frame = alpha158_frame(universe, window, pred_len, market=market)
    train = frame[(frame["date"] >= window.start_date) & (frame["date"] <= window.end_train_date)]
    normalizer = RobustZScore(train)
    discarded = discarded_samples(load_prices(universe), universe, pred_len, seq_len) if clean else None
    bounds = {
        "train": (window.start_date, window.end_train_date),
        "valid": (window.start_valid_date, window.end_valid_date),
        "test": (window.start_test_date, window.end_date),
    }
    return {split: DailyDataset(frame, seq_len, pred_len, *bounds[split], normalizer, split, discarded)
            for split in SPLITS}
