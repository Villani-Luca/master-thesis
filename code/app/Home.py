"""Streamlit entry point (THESIS_GUIDE.md Step 11). Run: streamlit run code/app/Home.py

Pages live in app/pages/ and are added step by step. The app only reads the results store and
calls xaifin; it contains no logic of its own.
"""

import streamlit as st

from xaifin.config import RESULTS_ROOT
from xaifin.models.registry import find_runs

st.set_page_config(page_title="XAI for Financial Models", layout="wide")
st.title("Applying Explainability Methods to Financial Models")
st.markdown(
    "Load trained forecasting models, inspect their feature attributions, and compare "
    "explanations across architectures, seeds, periods and universes."
)

runs = find_runs()

st.subheader("Trained runs")
if not runs.empty:
    st.dataframe(runs.drop(columns="path"), hide_index=True)
    st.caption("Open a run on the Models page.")
else:
    st.info(f"No trained runs found under {RESULTS_ROOT}.")
