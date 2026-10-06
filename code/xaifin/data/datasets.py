"""Daily cross-sectional datasets: one sample = one trading day = one DayBatch (THESIS_GUIDE.md §2.3).

Three ways of building the inputs, as in FinBench:

- DailyDataset (MASTER, MATCC): a ticker's window is its own last `seq_len` rows.
  Source: code/finbench/Regression/MASTER/load_dataset.py (CSVDataset) and train.py.
- FilledWindowDataset (FactorVAE): a ticker's window covers the last `seq_len` trading days, with
  the days it has no row filled. Source: code/finbench/Regression/FactorVAE/dataset.py
  (Qlib's TSDataSampler) and train.py.
- RowDataset (HIST, DiscoverPLF, FinFormer): no window, one Alpha360 row per sample (T = 1); the
  360 columns already cover 60 days. Sources: Regression/HIST/train.py (create_loaders, also used
  by DiscoverPLF) and Regression/FinFormer/train.py, load_dataset.py.

FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.

DailyDataset gives the same samples as FinBench's CSVDataset, in the same order: for each day, every ticker
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

from xaifin.config import CLEAN_DATA, rolling_window
from xaifin.data.loading import alpha_frame, load_concepts, load_market_cap, load_prices, load_sector_graph
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
    window = rolling_window(test_year)
    frame = alpha_frame(universe, window, pred_len, market=market)
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


def _ffill(idx: np.ndarray) -> np.ndarray:
    """Forward fill of row numbers along axis 0; -1 marks a missing row."""
    last = np.where(idx >= 0, np.arange(len(idx))[:, None], 0)
    np.maximum.accumulate(last, axis=0, out=last)
    return np.take_along_axis(idx, last, axis=0)


class FilledWindowDataset(Dataset):
    """The days of one split, as DayBatch, with FactorVAE's windows (Qlib TSDataSampler, 'ffill+bfill').

    The window of (ticker, day) covers the last `seq_len` days of the calendar (every date in the
    table), not the ticker's own last `seq_len` rows. Inside the window, a day without a row of the
    ticker takes its previous row (forward fill), or its next one at the start (back fill). Every
    ticker with a row on a day in [start_date, end_date] is a sample of that day; unlike
    DailyDataset nothing is cut at the end of the split, so the last training labels read prices of
    the validation period (as FinBench). `table` holds every date, with a Timestamp `date`.
    """

    def __init__(self, table: pd.DataFrame, seq_len: int, start_date: str, end_date: str,
                 feature_names: list[str], split: str = "train", discarded: pd.DataFrame | None = None):
        self.seq_len, self.split, self.feature_names = seq_len, split, feature_names
        table = table.sort_values(["date", "instrument"], ignore_index=True)
        self.values = torch.tensor(table[feature_names].to_numpy(np.float32))
        self.labels = torch.tensor(table["Label"].to_numpy(np.float32))

        calendar, day = np.unique(table["date"].to_numpy(), return_inverse=True)
        names, column = np.unique(table["instrument"].to_numpy(), return_inverse=True)
        self.grid = np.full((len(calendar), len(names)), -1)  # [day, ticker] -> row of the table
        self.grid[day, column] = np.arange(len(table))

        date = table["date"]
        rows = np.flatnonzero((date >= start_date) & (date <= end_date))
        if discarded is not None:
            samples = pd.MultiIndex.from_frame(table.loc[rows, ["instrument", "date"]])
            rows = rows[~samples.isin(pd.MultiIndex.from_frame(discarded[["instrument", "date"]]))]

        days, starts = np.unique(day[rows], return_index=True)  # rows are sorted by date, then ticker
        self.rows = np.split(rows, starts[1:])
        self.day_index = days
        self.columns = [column[r] for r in self.rows]  # per day: the grid column of each sample
        self.dates = [pd.Timestamp(calendar[d]) for d in days]
        self.tickers = [names[c].tolist() for c in self.columns]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> DayBatch:
        d = self.day_index[i]
        window = self.grid[max(d - self.seq_len + 1, 0): d + 1, self.columns[i]]  # [<=T, N] rows, -1 = none
        window = np.vstack([np.full((self.seq_len - len(window), window.shape[1]), -1), window])
        window = _ffill(_ffill(window)[::-1])[::-1]  # forward fill, then back fill
        return DayBatch(self.dates[i], self.tickers[i], self.values[torch.from_numpy(window.T.copy())],
                        self.labels[torch.from_numpy(self.rows[i])])


def factorvae_splits(universe: str, test_year: int, seq_len: int, clean: bool = CLEAN_DATA) -> dict[str, FilledWindowDataset]:
    """Train, valid and test datasets of FactorVAE for one rolling window, built as FinBench's train.py.

    Kept from FinBench (they change what the model learns, docs/model_notes.md):
    - issue 3: the label is the forward return over `seq_len` days, not `pred_len`;
    - issue 7: the first two features (KMID, KLEN) are not normalized, because FinBench takes
      `columns[2:-1]` after moving date and instrument into the index.
    The label is z-scored per day and clipped to [-3, 3] in the table; rows without a label are dropped.
    """
    window = rolling_window(test_year)
    frame = alpha_frame(universe, window, seq_len)
    frame["date"] = pd.to_datetime(frame["date"])
    features = [c for c in frame.columns if c not in ("instrument", "date", "Label")]
    train = frame[(frame["date"] >= window.start_date) & (frame["date"] <= window.end_train_date)]
    table = RobustZScore(train, columns=features[2:]).transform(frame)
    table = table[table["Label"].notna()].fillna(0)
    table["Label"] = table.groupby("date")["Label"].transform(lambda x: ((x - x.mean()) / x.std()).clip(-3, 3))
    discarded = discarded_samples(load_prices(universe), universe, seq_len, seq_len) if clean else None
    bounds = {
        "train": (window.start_date, window.end_train_date),
        "valid": (window.start_valid_date, window.end_valid_date),
        "test": (window.start_test_date, window.end_date),
    }
    return {split: FilledWindowDataset(table, seq_len, *bounds[split], features, split, discarded) for split in SPLITS}


def _drop_discarded(table: pd.DataFrame, discarded: pd.DataFrame | None) -> pd.DataFrame:
    """`table` without the samples of xaifin.data.quality.discarded_samples (None: unchanged)."""
    if discarded is None:
        return table
    samples = pd.MultiIndex.from_arrays([table["instrument"], pd.to_datetime(table["date"])])
    return table[~samples.isin(pd.MultiIndex.from_frame(discarded[["instrument", "date"]]))]


def _drop_last_rows(table: pd.DataFrame, n: int) -> pd.DataFrame:
    """`table` without the last `n` rows (in date order) of each ticker."""
    table = table.sort_values(["instrument", "date"], kind="stable")
    return table[table.groupby("instrument").cumcount(ascending=False) >= n]


class RowDataset(Dataset):
    """The days of one split, as DayBatch with T = 1: one row of `table` per sample (Alpha360 models).

    x is [N, 1, F] with the columns in `feature_names` order. Days with fewer than `min_stocks`
    samples are left out. extras, when given: 'market_value' [N] (column market_value of the
    table), 'stock2concept' [N, C] (the rows of `concepts` = (matrix, tickers) for the day's
    tickers), 'adjacency' [N, N] (the day's tickers in `graph` = (matrix, tickers)).
    """

    def __init__(self, table: pd.DataFrame, feature_names: list[str], split: str = "train", min_stocks: int = 1,
                 concepts: tuple[np.ndarray, list[str]] | None = None,
                 graph: tuple[np.ndarray, list[str]] | None = None):
        self.seq_len, self.split, self.feature_names = 1, split, feature_names
        table = table.sort_values(["date", "instrument"], ignore_index=True)
        self.values = torch.tensor(table[feature_names].to_numpy(np.float32))
        self.labels = torch.tensor(table["Label"].to_numpy(np.float32))
        self.market_value = (torch.tensor(table["market_value"].to_numpy(np.float32))
                             if "market_value" in table else None)

        date, instrument = table["date"].to_numpy(), table["instrument"].to_numpy()
        days, starts = np.unique(date, return_index=True)
        rows = np.split(np.arange(len(table)), starts[1:])
        keep = [i for i, r in enumerate(rows) if len(r) >= min_stocks]
        self.rows = [rows[i] for i in keep]
        self.dates = [pd.Timestamp(days[i]) for i in keep]
        self.tickers = [instrument[r].tolist() for r in self.rows]
        self.concepts = None if concepts is None else (torch.tensor(concepts[0], dtype=torch.float32),
                                                       {t: i for i, t in enumerate(concepts[1])})
        self.graph = None if graph is None else (torch.tensor(graph[0]), {t: i for i, t in enumerate(graph[1])})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> DayBatch:
        rows, tickers = torch.from_numpy(self.rows[i]), self.tickers[i]
        extras = {}
        if self.market_value is not None:
            extras["market_value"] = self.market_value[rows]
        if self.concepts is not None:
            matrix, position = self.concepts
            extras["stock2concept"] = matrix[[position[t] for t in tickers]]
        if self.graph is not None:
            matrix, position = self.graph
            index = torch.tensor([position[t] for t in tickers])
            extras["adjacency"] = matrix[index][:, index]
        return DayBatch(self.dates[i], tickers, self.values[rows][:, None, :], self.labels[rows], extras)


def hist_splits(universe: str, test_year: int, pred_len: int, constituents: str = "universe",
                clean: bool = CLEAN_DATA) -> dict[str, RowDataset]:
    """Train, valid and test datasets of HIST and DiscoverPLF, built as FinBench's create_loaders.

    Alpha360 rows; extras market_value (in $bn) and stock2concept. As FinBench: the robust z-score
    is fit on the training period; the last `pred_len` rows of each ticker are dropped twice, once
    over the whole table and once within each split; tickers missing from the concept matrix and
    stock-days without a market capitalization are dropped; valid and test start `pred_len` trading
    days early; days with a single stock are skipped. HIST reads its constituents from
    <universe>_constituents.csv ('universe'), DiscoverPLF from constituents/eodhd ('eodhd').
    """
    window = rolling_window(test_year)
    frame = alpha_frame(universe, window, pred_len, kind="alpha360", constituents=constituents)
    features = [c for c in frame.columns if c not in ("instrument", "date", "Label")]
    normalizer = RobustZScore(frame[(frame["date"] >= window.start_date) & (frame["date"] <= window.end_train_date)])
    table = normalizer.transform(_drop_last_rows(frame, pred_len)).fillna(0)
    table = table[(table["date"] >= window.start_date) & (table["date"] <= window.end_date)]
    concepts = load_concepts(universe)
    table = table[table["instrument"].isin(concepts[1])]

    calendar = np.sort(table["date"].unique())

    def early(start: str) -> str:  # the trading day pred_len days before `start`
        return calendar[max(0, np.searchsorted(calendar, start) - pred_len)]

    bounds = {
        "train": (window.start_date, window.end_train_date),
        "valid": (early(window.start_valid_date), window.end_valid_date),
        "test": (early(window.start_test_date), window.end_date),
    }
    market_cap = load_market_cap(universe)
    discarded = discarded_samples(load_prices(universe), universe, pred_len) if clean else None
    splits = {}
    for split, (start, end) in bounds.items():
        part = table[(table["date"] >= start) & (table["date"] <= end)]
        part = part.merge(market_cap[["instrument", "date", "market_value"]], on=["instrument", "date"], how="inner")
        part["market_value"] = part["market_value"].fillna(part["market_value"].mean())
        part = _drop_discarded(_drop_last_rows(part, pred_len), discarded)
        splits[split] = RowDataset(part, features, split, min_stocks=2, concepts=concepts)
    return splits


def finformer_splits(universe: str, test_year: int, pred_len: int, clean: bool = CLEAN_DATA) -> dict[str, RowDataset]:
    """Train, valid and test datasets of FinFormer, built as FinBench's train.py and CustomDataset.

    Alpha360 rows; extras adjacency (loading.load_sector_graph). As FinBench: the robust z-score is
    fit on the training period; valid and test start `pred_len` trading days early; the last
    `pred_len` days of each split are dropped, but not each ticker's last rows, so the last
    training labels read prices of the validation period; missing labels become 0 before the
    daily z-score of the label.
    """
    window = rolling_window(test_year)
    frame = alpha_frame(universe, window, pred_len, kind="alpha360")
    features = [c for c in frame.columns if c not in ("instrument", "date", "Label")]
    normalizer = RobustZScore(frame[(frame["date"] >= window.start_date) & (frame["date"] <= window.end_train_date)])
    calendar = np.sort(frame["date"].unique())
    bounds = {
        "train": (window.start_date, window.end_train_date),
        "valid": (calendar[np.argmax(calendar >= window.start_valid_date) - pred_len], window.end_valid_date),
        "test": (calendar[np.argmax(calendar >= window.start_test_date) - pred_len], window.end_date),
    }
    graph = load_sector_graph(universe)
    discarded = discarded_samples(load_prices(universe), universe, pred_len) if clean else None
    splits = {}
    for split, (start, end) in bounds.items():
        part = normalizer.transform(frame[(frame["date"] >= start) & (frame["date"] <= end)]).fillna(0)
        part = part[part["date"].isin(np.sort(part["date"].unique())[:-pred_len])]
        part = _drop_discarded(part, discarded).copy()
        part["Label"] = part["Label"].astype(np.float32)
        part["Label"] = part.groupby("date")["Label"].transform(lambda x: (x - x.mean()) / x.std())
        splits[split] = RowDataset(part, features, split, graph=graph)
    return splits
