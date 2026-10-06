"""Step 1 page: universes and data (THESIS_GUIDE.md).

Draws the cached summaries of xaifin.data.summary with the same figures as notebook
01_universes_and_data. Every chart has its table view next to it.
"""

import pandas as pd
import streamlit as st

from xaifin import viz
from xaifin.config import UNIVERSE_NAMES, UNIVERSES
from xaifin.data.features import catalog
from xaifin.data.summary import (
    REDUNDANT_ABS_CORR,
    model_groups_table,
    redundant_groups,
    redundant_pairs,
    summarize,
    summary_exists,
    universe_by_test_year,
    universe_table,
    volatility_by_year,
)

st.set_page_config(page_title="Data", layout="wide")
dark = st.context.theme.type == "dark"

st.title("Universes and data")
st.caption(
    "What the models see in each investment universe: membership, features, labels, feature redundancy "
    "and market volatility. Statistics cover the study period 2015-2024."
)



@st.cache_data(show_spinner=False)
def load_summary(universe: str) -> dict[str, pd.DataFrame]:
    return summarize(universe)


with st.expander("All universes and model groups"):
    cached = [u for u in UNIVERSES if summary_exists(u)]
    st.dataframe(
        universe_table({u: load_summary(u) for u in cached}),
        hide_index=True,
        column_config={
            "discarded": st.column_config.NumberColumn(format="percent"),
            "label_iqr": st.column_config.NumberColumn("label IQR", format="percent"),
            "share_positive": st.column_config.NumberColumn("positive labels", format="percent"),
            "index_volatility": st.column_config.NumberColumn("index volatility", format="percent"),
        },
    )
    st.caption(
        "Stock-days and labels cover all tickers in the files, 2015-2024. Discarded: share of stock-days removed by "
        "the data-quality filter. Label IQR: interquartile range of the 5-day return after the filter, mean over "
        "years. Index volatility: annualized, mean over 2015-2024."
        + ("" if len(cached) == len(UNIVERSES) else " Universes without a cached summary are not listed.")
    )
    st.dataframe(model_groups_table(), hide_index=True)

universe = st.selectbox("Universe", UNIVERSES, format_func=UNIVERSE_NAMES.get)
name = UNIVERSE_NAMES[universe]


if not summary_exists(universe):
    st.info(
        f"No cached summary for {name} yet. Computing it reads the price file and both feature files once: "
        "a few minutes for the core universes, longer for S&P 500 and STOXX Europe 600."
    )
    if st.button("Compute summary"):
        with st.spinner(f"Reading the {name} files..."):
            summarize(universe)
        load_summary.clear()
        st.rerun()
    st.stop()

s = load_summary(universe)
cat = catalog(universe)

coverage_tab, features_tab, labels_tab, redundancy_tab, volatility_tab = st.tabs(
    ["Coverage", "Features", "Labels", "Redundancy", "Market volatility"]
)

with coverage_tab:
    st.plotly_chart(viz.coverage_chart(s["membership"], s["coverage"], universe, dark), theme=None, width="stretch")
    st.markdown("**Universe at the start of each test year.** FinBench trains on the index members active then.")
    st.dataframe(universe_by_test_year(s), hide_index=True)

with features_tab:
    st.plotly_chart(viz.missing_chart(s["missing"], cat, universe, dark), theme=None, width="stretch")
    st.markdown(
        "**Feature catalog.** Alpha158 and Alpha360 overlap only on momentum, volume and candle features: "
        "compare the two model groups by *source* and *horizon*."
    )
    left, middle, right = st.columns(3)
    groups = left.multiselect("Group", sorted(cat["group"].unique()), default=sorted(cat["group"].unique()))
    families = middle.multiselect("Family", sorted(cat["family"].unique()), default=sorted(cat["family"].unique()))
    horizons = right.multiselect("Horizon", ["short", "medium", "long"], default=["short", "medium", "long"])
    table = cat.merge(s["missing"], on=["group", "feature"], how="left")
    table = table[table["group"].isin(groups) & table["family"].isin(families) & table["horizon"].isin(horizons)]
    st.dataframe(
        table[["group", "feature", "family", "source", "horizon", "window", "lag", "constant", "missing_share", "description"]],
        hide_index=True,
        column_config={"missing_share": st.column_config.NumberColumn("missing", format="percent")},
    )
    with st.expander("Counts by family and group"):
        st.dataframe(pd.crosstab(cat["family"], cat["group"], margins=True))

with labels_tab:
    st.plotly_chart(viz.label_chart(s["label_stats_clean"], universe, dark), theme=None, width="stretch")
    st.markdown(
        "The label is FinBench's forward return on the adjusted close, shown before the daily cross-sectional "
        "z-score the models apply and after the data-quality filter below."
    )
    stats = s["label_stats_clean"][["year", "n", "mean", "std", "p05", "median", "p95", "share_positive"]]
    stats = stats.assign(std_before_filter=s["label_stats"]["std"].to_numpy())
    st.dataframe(stats.style.format({c: "{:.2%}" for c in stats.columns if c not in ("year", "n")}), hide_index=True)
    st.markdown(
        "**Extreme labels.** Instruments whose price doubles or halves within 5 days. Some are real moves, many are "
        "data errors (bad prices, wrong ticker mappings, a delisting recorded as 0); only those with test years "
        "reach the models."
    )
    st.dataframe(
        s["extreme_labels"],
        hide_index=True,
        column_config={
            "min_label": st.column_config.NumberColumn(format="percent"),
            "max_label": st.column_config.NumberColumn(format="percent"),
        },
    )
    rows = s["meta"]["rows_alpha158"].iloc[0]
    st.markdown(
        f"**Discarded samples: {s['discarded']['samples'].sum():,} of {rows:,} "
        f"({s['discarded']['samples'].sum() / rows:.2%}).** The filter (`xaifin/data/quality.py`) removes a sample "
        "when its input or label reads a ticker of the wrong company, a stale price (20+ identical days), a price "
        "below 0.01 or a one-day jump beyond ×3."
    )
    st.dataframe(s["discarded"], hide_index=True)

with redundancy_tab:
    groups_table = redundant_groups(s["alpha158_clusters"])
    pairs = redundant_pairs(s["alpha158_corr"])
    a, b, c = st.columns(3)
    a.metric("Independent groups", s["alpha158_clusters"]["cluster"].nunique(), help="Out of 157 Alpha158 features")
    b.metric("Features in near-duplicate groups", int(groups_table["size"].sum()))
    c.metric(f"Pairs with |ρ| ≥ {REDUNDANT_ABS_CORR}", len(pairs))
    st.plotly_chart(
        viz.correlation_chart(s["alpha158_corr"], s["alpha158_clusters"], universe, dark), theme=None, width="stretch"
    )
    st.markdown(
        "Near-duplicate features split or blur attribution credit (SHAP, permutation importance), and must move "
        "together in counterfactuals."
    )
    left, right = st.columns(2)
    left.markdown("**Near-duplicate groups**")
    left.dataframe(groups_table, hide_index=True)
    right.markdown("**Most correlated pairs**")
    right.dataframe(pairs, hide_index=True, column_config={"abs_spearman": st.column_config.NumberColumn("|ρ|", format="%.3f")})

with volatility_tab:
    st.plotly_chart(viz.volatility_chart(s["market_volatility"], universe, dark), theme=None, width="stretch")
    st.markdown("**Mean annualized volatility per year**")
    st.dataframe(volatility_by_year(s["market_volatility"]).style.format("{:.1%}"))
