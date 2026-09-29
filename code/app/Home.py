"""Streamlit entry point (THESIS_GUIDE.md Step 11). Run: streamlit run code/app/Home.py

Pages live in app/pages/ and are added step by step. The app only reads the results store and
calls xaifin; it contains no logic of its own.
"""

import pandas as pd
import streamlit as st

from xaifin.config import RESULTS_ROOT

st.set_page_config(page_title="XAI for Financial Models", layout="wide")
st.title("Applying Explainability Methods to Financial Models")
st.markdown(
    "Load trained forecasting models, inspect their feature attributions, and compare "
    "explanations across architectures, seeds, periods and universes."
)

# Results layout: <Task>/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<YEAR>/model.pth
runs = [
    dict(zip(["task", "model", "universe", "config", "seed", "year"], p.relative_to(RESULTS_ROOT).parts[:6]))
    | {"has_metrics": (p.parent / "metrics.json").exists()}
    for p in sorted(RESULTS_ROOT.glob("*/*/*/*/*/*/model.pth"))
]

st.subheader("Trained runs")
if runs:
    st.dataframe(pd.DataFrame(runs), hide_index=True)
else:
    st.info(f"No trained runs found under {RESULTS_ROOT}.")
