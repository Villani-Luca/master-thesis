"""Data-quality filter: the samples whose prices cannot be trusted (notebook 01, section 6).

The price files contain errors that reach the models: stale prices (SAABY.US stays at 992,237 for
five years), prices at or near zero (DISH.US after its merger, ORRON.ST at 0.0002), one-day jumps
that no index member makes (VIV.PA at 1% of its price for a day, unadjusted corporate actions,
quotes after a bank seizure) and tickers whose prices belong to another company (DJI's PRG.US).
FinBench trains on them as they are; the thesis discards them (decided 2026-09-29), in training,
validation and test alike.

The filter works on samples, not on rows of the files: the sample of stock i on day t is discarded
when a problem falls in the days it reads, from the first price behind its input features to the
end of its label. Deleting the rows before building the input sequences would instead join the days
before and after a gap. The Step 2 datasets apply it when config.CLEAN_DATA is True; the
equivalence check against FinBench turns it off.
"""

import numpy as np
import pandas as pd

from xaifin.data.features import WINDOWS

STALE_DAYS = 20  # the same adjusted close on at least this many consecutive trading days
MIN_PRICE = 0.01  # adjusted closes below this are errors or quotes after a delisting
MAX_DAILY_MOVE = 3.0  # a price that triples, or falls to a third, from one day to the next
# Prices read by the features of one day: the 60-day windows of Alpha158 plus the previous close
# (ROC60, CNTP60, ...); Alpha360 reads 60 days.
FEATURE_LOOKBACK = max(WINDOWS) + 1

# Tickers whose prices belong to another company than the index member (notebook 01, Observations 4).
WRONG_COMPANY = {
    "dji": {"PRG.US": "listed as Procter & Gamble, prices of PROG Holdings"},
    "sxxp": {
        "TEN.US": "listed as Tenaris, prices of Tsakos Energy Navigation",
        "TOM.F": "listed as Tomra Systems, prices of Toyota Motor",
    },
}
REASONS = ["wrong_company", "stale_price", "min_price", "price_jump"]


def _in_window(flag: pd.Series, ticker: pd.Series, before: int, after: int) -> pd.Series:
    """True where `flag` holds on any of the rows t - before + 1 ... t + after of the same ticker."""
    seen = flag.astype(int).groupby(ticker).cumsum()
    ahead = seen.groupby(ticker).shift(-after).fillna(seen.groupby(ticker).transform("last"))
    behind = seen.groupby(ticker).shift(before, fill_value=0)
    return ahead - behind > 0


def discarded_samples(prices: pd.DataFrame, universe: str, pred_len: int, seq_len: int = 1) -> pd.DataFrame:
    """Samples (instrument, date) to discard, with the first rule that applies:

    - `wrong_company`: the ticker is in WRONG_COMPANY;
    - `stale_price`: a price that belongs to a run of STALE_DAYS or more identical prices,
    - `min_price`: a price below MIN_PRICE,
    - `price_jump`: a move beyond MAX_DAILY_MOVE from one day to the next,

    anywhere in the days the sample reads: `seq_len` days of input features, each reading
    FEATURE_LOOKBACK prices, and the `pred_len` days of its label. `prices` needs instrument, date
    and adj_close (loading.load_prices).
    """
    p = prices[["instrument", "date", "adj_close"]].sort_values(["instrument", "date"], ignore_index=True)
    ticker, price = p["instrument"], p["adj_close"]

    run = price.groupby(ticker).diff().ne(0).groupby(ticker).cumsum()
    log_price = np.log(price.where(price > 0))
    flags = {
        "stale_price": price.groupby([ticker, run]).transform("size") >= STALE_DAYS,
        "min_price": price < MIN_PRICE,
        "price_jump": log_price.groupby(ticker).diff().abs() > np.log(MAX_DAILY_MOVE),
    }
    before = FEATURE_LOOKBACK + seq_len - 1
    rules = {"wrong_company": ticker.isin(WRONG_COMPANY.get(universe, {}))}
    rules |= {name: _in_window(flag, ticker, before, pred_len) for name, flag in flags.items()}

    reason = np.select(list(rules.values()), list(rules), default="")
    keep = reason != ""
    return p.loc[keep, ["instrument", "date"]].assign(reason=reason[keep]).reset_index(drop=True)


def drop_discarded(samples: pd.DataFrame, discarded: pd.DataFrame) -> pd.DataFrame:
    """`samples` (with instrument and date columns) without the discarded ones."""
    index = pd.MultiIndex.from_frame(samples[["instrument", "date"]])
    return samples[~index.isin(pd.MultiIndex.from_frame(discarded[["instrument", "date"]]))]
