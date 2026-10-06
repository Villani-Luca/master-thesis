"""Project-wide paths and experimental protocol (THESIS_GUIDE.md §2.4-2.5)."""

from pathlib import Path
from typing import NamedTuple

CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = CODE_ROOT / "data"
RESULTS_ROOT = CODE_ROOT / "results"

# Universes available in code/data/; the thesis focuses on the smaller ones (outline, "Link utili").
UNIVERSES = ["dji", "nasdaq100", "sp500", "sx5e", "sxxp"]
CORE_UNIVERSES = ["dji", "nasdaq100", "sx5e"]
UNIVERSE_NAMES = {
    "dji": "DJI",
    "nasdaq100": "NASDAQ-100",
    "sp500": "S&P 500",
    "sx5e": "EURO STOXX 50",
    "sxxp": "STOXX Europe 600",
}

# Universe -> nation of its market gate features (<nation>_market.csv), as in FinBench MATCC/train.py:398.
NATION = {"dji": "us", "nasdaq100": "us", "sp500": "us", "sx5e": "eu", "sxxp": "eu"}

# Universe -> its own index among the market indexes of <nation>_market.csv.
BENCHMARK_INDEX = {"dji": "DJI.INDX", "nasdaq100": "NDX.INDX", "sp500": "GSPC.INDX", "sx5e": "SX5E.INDX", "sxxp": "SXXP.INDX"}

# Directly comparable models: same features, same protocol (docs/model_groups.md).
MODEL_GROUPS = {
    "alpha158": ["MASTER", "MATCC", "FactorVAE"],
    "alpha360": ["HIST", "DiscoverPLF", "FinFormer"],
}

# Lookback T and horizon L, as (seq_len, pred_len), from the FinBench paper §4.1: daily, weekly, monthly.
SL_PL_CONFIGS = [(5, 1), (20, 5), (60, 20)]
DEFAULT_SL_PL = (20, 5)
ALPHA360_SEQ_LEN = 1  # Alpha360 models: one 360-vector per day, which already covers 60 days.

# Discard the samples with unusable prices (xaifin.data.quality), in training, validation and test.
# Off only to reproduce FinBench exactly (Step 2 equivalence check).
CLEAN_DATA = True

# Random seeds of the FinBench paper §4.1.
SEEDS = [0, 5, 42]

# Long-only top-k portfolio size. FinBench Evaluation/evaluation.py defaults to 10; DJI has only
# ~30 stocks, so 10 would hold a third of the universe.
TOP_K = {"dji": 5, "nasdaq100": 10, "sp500": 10, "sx5e": 10, "sxxp": 10}


class RollingWindow(NamedTuple):
    """Date boundaries of one rolling phase, named like the arguments of FinBench's train.py scripts."""

    start_date: str
    end_train_date: str
    start_valid_date: str
    end_valid_date: str
    start_test_date: str
    end_date: str


def rolling_window(test_year: int) -> RollingWindow:
    """FinBench rolling protocol (paper §4.1): 4 years of training, 1 of validation, 1 of test."""
    return RollingWindow(
        start_date=f"{test_year - 5}-01-01",
        end_train_date=f"{test_year - 2}-12-31",
        start_valid_date=f"{test_year - 1}-01-01",
        end_valid_date=f"{test_year - 1}-12-31",
        start_test_date=f"{test_year}-01-01",
        end_date=f"{test_year}-12-31",
    )


TEST_YEARS = [2020, 2021, 2022, 2023, 2024]
ROLLING_WINDOWS = {year: rolling_window(year) for year in TEST_YEARS}

# Period covered by all rolling windows: first training day to last test day (2015-2024).
STUDY_START = ROLLING_WINDOWS[TEST_YEARS[0]].start_date
STUDY_END = ROLLING_WINDOWS[TEST_YEARS[-1]].end_date
