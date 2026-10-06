"""Step 4 page: long-only top-k portfolios (THESIS_GUIDE.md).

Every week (L = 5 trading days) the portfolio buys the k stocks with the highest predicted score, in
equal parts, and holds them until the next rebalance. Compared with two benchmarks: all the stocks
the model ranks, in equal parts, and the index.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from xaifin import viz
from xaifin.config import UNIVERSE_NAMES
from xaifin.portfolio.backtest import find_portfolios

st.set_page_config(page_title="Portfolio", layout="wide")
dark = st.context.theme.type == "dark"

st.title("Top-k portfolios")
st.caption(
    "Every week the portfolio buys the k stocks with the highest predicted score, in equal parts, and holds them "
    "until the next rebalance (buying at the open). Benchmarks: all the stocks the model ranks, in equal parts "
    "('equal weight'), and the index."
)


@st.cache_data(show_spinner=False)
def portfolios() -> pd.DataFrame:
    return find_portfolios()


@st.cache_data(show_spinner=False)
def load(path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    return (json.loads((Path(path) / "metrics.json").read_text()),
            pd.read_parquet(Path(path) / "returns.parquet"),
            pd.read_parquet(Path(path) / "holdings.parquet"))


table = portfolios()
if table.empty:
    st.info("No portfolios yet: run code/scripts/run_portfolio.py once a model has all its test years trained.")
    st.stop()

left, middle, right = st.columns(3)
universe = left.selectbox("Universe", sorted(table["universe"].unique()), format_func=UNIVERSE_NAMES.get)
choice = table[table["universe"] == universe]
seed = middle.selectbox("Seed", sorted(choice["seed"].unique()))
choice = choice[choice["seed"] == seed]
k = right.selectbox("Stocks held (k)", sorted(choice["k"].unique()))
choice = choice[choice["k"] == k]
order = [m for m in viz.MODEL_ORDER if m in set(choice["model"])]
models = st.multiselect("Models", order, default=order)
choice = choice.set_index("model").loc[models]
if choice.empty:
    st.stop()

loaded = {m: load(str(row["path"])) for m, row in choice.iterrows()}
first = next(iter(loaded.values()))
curves = pd.DataFrame({m: (1 + r.set_index("date")["portfolio"]).cumprod() for m, (_, r, _) in loaded.items()})
benchmarks = first[1].set_index("date")
curves["equal weight"] = (1 + benchmarks["equal_weight"]).cumprod()
curves["index"] = (1 + benchmarks["index"]).cumprod()
years = first[0]["setup"]["test_years"]
name = UNIVERSE_NAMES[universe]
st.plotly_chart(viz.equity_chart(curves, f"{name}: top-{k} portfolios, {years[0]}-{years[-1]}", dark), theme=None,
                width="stretch")

rows = {m: m_[0]["portfolio"] for m, m_ in loaded.items()}
rows["equal weight"], rows["index"] = first[0]["equal_weight"], first[0]["index"]
metrics = pd.DataFrame(rows).T[["CAGR", "Sharpe", "Sortino", "max_drawdown", "volatility", "turnover"]]
st.dataframe(metrics.style.format({"CAGR": "{:+.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "max_drawdown": "{:.1%}",
                                   "volatility": "{:.1%}", "turnover": "{:.0%}"}, na_rep="–"))
st.caption("Turnover: the average share of the portfolio replaced at each weekly rebalance. Metrics as quantstats (rf = 0).")
yearly = pd.DataFrame({m: m_[0]["yearly_returns"]["portfolio"] for m, m_ in loaded.items()})
yearly["equal weight"], yearly["index"] = first[0]["yearly_returns"]["equal_weight"], first[0]["yearly_returns"]["index"]
st.markdown("**Return per year**")
st.dataframe(yearly.T.style.format("{:+.1%}"))

st.subheader("One model's holdings")
model = st.selectbox("Model", models)
holdings = loaded[model][2]
st.plotly_chart(viz.holdings_map(holdings, f"{name}: stocks held by the {model} top-{k} portfolio", dark), theme=None,
                width="stretch")
st.plotly_chart(viz.margin_histogram(holdings, f"{name}, {model}: how close the held stocks are to dropping out", dark),
                theme=None, width="stretch")
st.caption(
    "Margin: how much a held stock's score could fall before another stock overtakes it, in standard deviations of "
    "that day's scores. Small margins mean the portfolio would change with small changes in the predictions."
)
dates = sorted(holdings["start"].unique())
day = st.select_slider("Period", options=dates, value=dates[0], format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
st.dataframe(holdings[holdings["start"] == day].sort_values("rank").drop(columns=["start", "end"]), hide_index=True)
