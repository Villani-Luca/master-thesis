"""Daily returns and performance of top-k portfolios (THESIS_GUIDE.md Step 4).

Returns as FinBench's portfolio_daily_returns (Evaluation/portfolio/returns.py): within each holding
period, each held stock earns its close-to-close return on adjusted prices, except on its first day,
where it earns the open-to-close return (the portfolio buys at the open); a stock without a price on
a day earns 0; the portfolio return is the weighted sum. Metrics as FinBench's quantstats report
(Evaluation/quantstats/stats.py, rf = 0, compounded): checked against it in notebook 04.
FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.

Two benchmarks on the same periods: the equal-weight portfolio of all candidate stocks (the
universe the model ranks) and the index (daily return of the universe's index, market file).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xaifin.config import BENCHMARK_INDEX, RESULTS_ROOT, TOP_K
from xaifin.data.loading import load_market, load_prices

TRADING_DAYS = 252


def asset_returns(universe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (date x ticker) close-to-close and open-to-close returns on adjusted prices.

    As FinBench: per ticker, rows without both prices are dropped, then pct_change over the ticker's
    own consecutive rows.
    """
    p = load_prices(universe, ("instrument", "date", "adj_close", "adj_open")).dropna()
    p = p.sort_values(["instrument", "date"])
    p["returns"] = p.groupby("instrument")["adj_close"].pct_change()
    p["open_to_close"] = (p["adj_close"] - p["adj_open"]) / p["adj_open"]
    p = p.dropna(subset=["returns"])
    wide = lambda col: p.pivot(index="date", columns="instrument", values=col).sort_index()
    return wide("returns"), wide("open_to_close")


def portfolio_returns(holdings: pd.DataFrame, returns: pd.DataFrame, open_to_close: pd.DataFrame) -> pd.Series:
    """Daily returns of the portfolio described by `holdings` (topk.topk_holdings; rows with weight > 0)."""
    held = holdings[holdings["weight"] > 0]
    parts = []
    for (start, end), g in held.groupby(["start", "end"], sort=True):
        tickers, weights = g["ticker"].tolist(), g["weight"].to_numpy()
        present = [t for t in tickers if t in returns.columns]
        window = returns.loc[(returns.index >= start) & (returns.index < end), present]
        window = window.dropna(how="all")
        if window.empty:
            parts.append(pd.Series(0.0, index=pd.date_range(start, end, freq="B")[:-1]))
            continue
        window = window.copy()
        for t in present:  # first day of each stock in the period: bought at the open
            first = window[t].first_valid_index()
            if first is not None:
                window.loc[first, t] = open_to_close.loc[first, t]
        window = window.reindex(columns=tickers).fillna(0.0)
        parts.append(pd.Series(window.to_numpy() @ weights, index=window.index))
    return pd.concat(parts).sort_index()


def equal_weight_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """The benchmark: every candidate stock of each period, with equal weights."""
    out = holdings.copy()
    out["weight"] = 1.0 / out.groupby("start")["ticker"].transform("size")
    return out


def index_returns(universe: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Daily return of the universe's index on `dates`, from the market file (price_change_<index>)."""
    market = load_market(universe)
    series = market.set_index(pd.to_datetime(market["date"]))[f"price_change_{BENCHMARK_INDEX[universe]}"]
    return series.reindex(dates).fillna(0.0)


def performance(returns: pd.Series) -> dict:
    """quantstats' CAGR, Sharpe, Sortino, max drawdown and volatility (rf = 0, compounded, 252 days)."""
    r = returns.astype(float)
    total = float((1 + r).prod() - 1)
    years = (r.index[-1] - r.index[0]).days / 365
    prices = (1 + r).cumprod()
    downside = np.sqrt((r[r < 0] ** 2).sum() / len(r))
    return {
        "total_return": total,
        "CAGR": float(abs(total + 1.0) ** (1.0 / years) - 1),
        "Sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "Sortino": float(r.mean() / downside * np.sqrt(TRADING_DAYS)),
        "max_drawdown": float((prices / prices.cummax()).min() - 1),
        "volatility": float(r.std() * np.sqrt(TRADING_DAYS)),
        "days": int(len(r)),
    }


def turnover(holdings: pd.DataFrame) -> float:
    """Mean one-way turnover per rebalance: half the sum of |weight changes| between consecutive periods."""
    w = holdings.pivot_table(index="start", columns="ticker", values="weight", fill_value=0.0).sort_index()
    return float((w.diff().abs().sum(axis=1) / 2).iloc[1:].mean())


def yearly_returns(returns: pd.Series) -> pd.Series:
    """Compounded return per calendar year (FinBench's EOY returns)."""
    return returns.groupby(returns.index.year).apply(lambda r: (1 + r).prod() - 1)


def portfolio_dir(model: str, universe: str, seq_len: int, pred_len: int, seed: int, k: int) -> Path:
    """results/portfolio/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/top<k>/ (THESIS_GUIDE.md §2.2)."""
    return RESULTS_ROOT / "portfolio" / model / universe / f"sl{seq_len}_pl{pred_len}" / f"seed{seed}" / f"top{k}"


def run_portfolio(run_dirs: list[Path], model: str, universe: str, seq_len: int, pred_len: int, seed: int,
                  k: int | None = None, returns: tuple | None = None) -> dict:
    """Holdings, daily returns and metrics of one model's top-k portfolio over its test years; saved to portfolio_dir.

    `run_dirs`: the trained runs of consecutive test years (one per year). `returns`: the output of
    asset_returns(universe), to reuse it across models. Writes holdings.parquet (every candidate,
    held or not), returns.parquet (portfolio, equal-weight benchmark, index) and metrics.json.
    """
    from xaifin.portfolio.topk import load_predictions, rebalance_periods, topk_holdings

    k = int(k or TOP_K[universe])
    seq_len, pred_len, seed = int(seq_len), int(pred_len), int(seed)
    years = sorted(int(Path(d).name[1:]) for d in run_dirs)
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError(f"test years must be consecutive, got {years}")
    predictions = load_predictions(run_dirs)
    periods = rebalance_periods(f"{years[0]}-01-01", f"{years[-1]}-12-31", pred_len)
    holdings = topk_holdings(predictions, periods, k)
    rets, otc = returns or asset_returns(universe)

    daily = pd.DataFrame({"portfolio": portfolio_returns(holdings, rets, otc)})
    daily["equal_weight"] = portfolio_returns(equal_weight_holdings(holdings), rets, otc).reindex(daily.index).fillna(0.0)
    daily["index"] = index_returns(universe, daily.index)
    metrics = {name: performance(daily[name]) for name in daily.columns}
    metrics["portfolio"]["turnover"] = turnover(holdings)
    metrics["portfolio"]["excess_CAGR_vs_equal_weight"] = metrics["portfolio"]["CAGR"] - metrics["equal_weight"]["CAGR"]
    metrics["yearly_returns"] = {name: {str(y): float(v) for y, v in yearly_returns(daily[name]).items()}
                                 for name in daily.columns}
    metrics["setup"] = {"model": model, "universe": universe, "seq_len": seq_len, "pred_len": pred_len, "seed": seed,
                        "k": k, "test_years": years, "periods": len(periods),
                        "first_day": str(daily.index[0].date()), "last_day": str(daily.index[-1].date())}

    out = portfolio_dir(model, universe, seq_len, pred_len, seed, k)
    out.mkdir(parents=True, exist_ok=True)
    holdings.to_parquet(out / "holdings.parquet", index=False)
    daily.rename_axis("date").reset_index().to_parquet(out / "returns.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def expected_years(universe: str) -> list[int]:
    """Test years with a universe (no 2020 for the EU universes: their membership history starts 2020-09-30)."""
    from xaifin.config import TEST_YEARS
    return [y for y in TEST_YEARS if not (universe in ("sx5e", "sxxp") and y == 2020)]


def run_portfolio_grid(results: pd.DataFrame, k: int | None = None, partial: bool = False, force: bool = False,
                       log=print) -> None:
    """run_portfolio for every (model, universe, T, L, seed) of `results` (training.rolling.collect_results).

    Only groups whose test years are all trained, unless `partial` (then the consecutive years from
    the first). Groups with a metrics.json are skipped unless `force`.
    """
    cache = {}
    for (model, universe, seq_len, pred_len, seed), g in results.groupby(["model", "universe", "seq_len", "pred_len", "seed"]):
        years = sorted(g["test_year"])
        wanted = expected_years(universe)
        if years != wanted and not partial:
            log(f"{model} {universe} seed {seed}: {len(years)}/{len(wanted)} test years trained, skipped")
            continue
        run = [years[0]]
        while run[-1] + 1 in years:
            run.append(run[-1] + 1)
        kk = k or TOP_K[universe]
        if (portfolio_dir(model, universe, seq_len, pred_len, seed, kk) / "metrics.json").exists() and not force:
            log(f"{model} {universe} seed {seed}: done already")
            continue
        if universe not in cache:
            cache[universe] = asset_returns(universe)
        dirs = [g.loc[g["test_year"] == y, "path"].iloc[0] for y in run]
        m = run_portfolio(dirs, model, universe, seq_len, pred_len, seed, kk, cache[universe])
        log(f"{model} {universe} seed {seed} top{kk} {run[0]}-{run[-1]}: CAGR {m['portfolio']['CAGR']:+.2%} "
            f"(equal weight {m['equal_weight']['CAGR']:+.2%}, index {m['index']['CAGR']:+.2%}), "
            f"Sharpe {m['portfolio']['Sharpe']:.2f}, turnover {m['portfolio']['turnover']:.0%}")


def find_portfolios(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """One row per saved portfolio: model, universe, seq_len, pred_len, seed, k, test_years, path."""
    rows = []
    for metrics_file in sorted(Path(results_root).glob("portfolio/*/*/*/*/*/metrics.json")):
        setup = json.loads(metrics_file.read_text())["setup"]
        rows.append({**{k: setup[k] for k in ("model", "universe", "seq_len", "pred_len", "seed", "k")},
                     "test_years": f"{setup['test_years'][0]}-{setup['test_years'][-1]}", "path": metrics_file.parent})
    return pd.DataFrame(rows, columns=["model", "universe", "seq_len", "pred_len", "seed", "k", "test_years", "path"])
