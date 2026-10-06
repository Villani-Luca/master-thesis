"""Step 2 page: trained models (THESIS_GUIDE.md).

Pick a run of the results store, then see its configuration, training curve and test metrics, and
the predictions of its checkpoint, recomputed through the model's adapter (xaifin.models.registry).
"""

import json

import pandas as pd
import streamlit as st

from xaifin import viz
from xaifin.config import TOP_K, UNIVERSE_NAMES
from xaifin.models.registry import find_runs, load_run, run_config
from xaifin.training.metrics import daily_ic, summary
from xaifin.training.rolling import collect_results

st.set_page_config(page_title="Models", layout="wide")
dark = st.context.theme.type == "dark"

st.title("Trained models")
st.caption(
    "Every trained run of the results store: its configuration, training curve and test metrics, and the "
    "predictions of its checkpoint, recomputed through the model's adapter."
)


@st.cache_data(show_spinner=False)
def runs() -> pd.DataFrame:
    return find_runs()


@st.cache_data(show_spinner=False)
def results() -> pd.DataFrame:
    return collect_results()


@st.cache_resource(show_spinner=False)
def adapter_of(run_dir: str):
    return load_run(run_dir)


@st.cache_data(show_spinner=False)
def test_predictions(run_dir: str) -> tuple[pd.DataFrame, dict]:
    """One row per test stock-day (date, ticker, prediction, label), and the test metrics."""
    adapter = adapter_of(run_dir)
    frames, preds, labels = [], [], []
    for batch in adapter.day_batches("test"):
        pred, label = adapter.predict(batch), adapter.target(batch).numpy()
        preds.append(pred)
        labels.append(label)
        frames.append(pd.DataFrame({"date": batch.date, "ticker": batch.tickers, "prediction": pred, "label": label}))
    return pd.concat(frames, ignore_index=True), summary(preds, labels)


table = runs()
if table.empty:
    st.info("No trained runs yet. Runs appear here once Step 3 has saved them under results/Regression/.")
    st.stop()

with st.expander(f"All runs ({len(table)})"):
    st.dataframe(table.drop(columns="path"), hide_index=True)

st.subheader("Test metrics: model × test year")
done = results()
if done.empty:
    st.info("No finished run has a config.json yet.")
else:
    left, middle, right = st.columns(3)
    grid_universe = left.selectbox("Universe", sorted(done["universe"].unique()), format_func=UNIVERSE_NAMES.get,
                                  key="grid_universe")
    metric = middle.selectbox("Metric", ["RankIC", "IC", "RankICIR", "ICIR", "IC_pooled"], key="grid_metric")
    horizons = sorted(done[["seq_len", "pred_len"]].drop_duplicates().itertuples(index=False, name=None))
    seq_len, pred_len = right.selectbox("T, L", horizons, format_func=lambda h: f"T={h[0]}, L={h[1]}")
    cells = done[(done["universe"] == grid_universe) & (done["seq_len"] == seq_len) & (done["pred_len"] == pred_len)]
    grid = cells.groupby(["model", "test_year"])[metric].agg(["mean", "std", "count"]).reset_index()
    grid = grid.rename(columns={"mean": metric, "std": f"{metric} std", "count": "seeds"})
    title = f"{UNIVERSE_NAMES[grid_universe]}: test {metric} by model and test year"
    st.plotly_chart(viz.metric_heatmap(grid, metric, title, dark), theme=None, width="stretch")
    st.caption(
        f"Mean over the seeds trained so far (column 'seeds' below). Blue: {metric} above 0, red: below. "
        "Each model keeps its weights with FinBench's own rule (THESIS_GUIDE.md Step 3)."
    )
    st.dataframe(grid.pivot(index="model", columns="test_year", values=metric).style.format("{:+.4f}", na_rep="–"))
    with st.expander("Cells with seed counts and standard deviations"):
        st.dataframe(grid, hide_index=True)

st.subheader("One run")

columns = st.columns(5)
choice = table
for column, (field, label) in zip(columns, [("model", "Model"), ("universe", "Universe"), ("seq_len", "Lookback T"),
                                            ("seed", "Seed"), ("test_year", "Test year")]):
    options = sorted(choice[field].unique())
    value = column.selectbox(label, options, format_func=UNIVERSE_NAMES.get if field == "universe" else str)
    choice = choice[choice[field] == value]
run = choice.iloc[0]
run_dir = str(run["path"])
cfg, has_config = run_config(run_dir)
name = f"{cfg.model}, {UNIVERSE_NAMES[cfg.universe]}, test year {cfg.test_year}, seed {cfg.seed}"

overview_tab, training_tab, predictions_tab = st.tabs(["Overview", "Training", "Test predictions"])

with overview_tab:
    stored = json.loads((run["path"] / "metrics.json").read_text()) if run["metrics"] else {}
    if stored:
        cards = st.columns(4)
        for card, key in zip(cards, ["IC", "RankIC", "MSE", "R2"]):
            if key in stored:
                card.metric(f"Test {key}", f"{stored[key]:.4f}")
        st.caption("Stored test metrics (metrics.json). IC and RankIC are means of the daily cross-sectional values.")
    else:
        st.info("This run has no metrics.json.")
    st.markdown("**Configuration**")
    adapter_class = adapter_of(run_dir).__class__
    fields = {"model": cfg.model, "group": adapter_class.group, "universe": cfg.universe, "test year": cfg.test_year,
              "seed": cfg.seed, "lookback T": cfg.seq_len, "horizon L": cfg.pred_len,
              "data-quality filter": "on" if cfg.clean else "off"}
    st.dataframe(pd.DataFrame({"value": {k: str(v) for k, v in fields.items()}}))
    hparams = {**adapter_class.HPARAMS, **cfg.hparams}
    st.dataframe(pd.DataFrame({"value": {k: str(v) for k, v in hparams.items()}}), height=260)
    if not has_config:
        st.caption(
            "This run has no config.json (it predates ModelAdapter.save): the configuration is read from the folder "
            "names, with FinBench's hyper-parameters and the data-quality filter off, as it was trained."
        )

with training_tab:
    if run["history"]:
        history = pd.read_csv(run["path"] / "train_history.csv")
        kept = int(history.loc[history["RankIC"].idxmax(), "epoch"]) if "RankIC" in history else None
        st.plotly_chart(viz.training_chart(history, f"{name}: training", kept, dark), theme=None, width="stretch")
        st.caption("Kept epoch: the best validation RankIC, the rule of these runs.")
        st.dataframe(history, hide_index=True)
    else:
        st.info("This run has no train_history.csv.")

with predictions_tab:
    with st.spinner("Loading the checkpoint and the test data..."):
        predictions, metrics = test_predictions(run_dir)
    st.caption(
        f"Predictions of the checkpoint on the {predictions['date'].nunique()} test days, recomputed with the "
        f"{cfg.model} adapter on the same data as in training."
    )
    compare = pd.DataFrame({"recomputed": metrics, "stored": stored}).loc[["IC", "RankIC", "ICIR", "RankICIR", "MSE"]]
    st.dataframe(compare.style.format("{:.4f}", na_rep="–"))

    daily = predictions.groupby("date").apply(
        lambda d: pd.Series(daily_ic(d["prediction"].to_numpy(), d["label"].to_numpy()), index=["IC", "RankIC"]),
        include_groups=False).reset_index()
    st.plotly_chart(viz.daily_ic_chart(daily, f"{name}: daily RankIC on the test year", dark=dark), theme=None,
                    width="stretch")

    dates = sorted(predictions["date"].unique())
    day = st.select_slider("Day", options=dates, value=dates[0], format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    k = TOP_K[cfg.universe]
    selected = predictions[predictions["date"] == day].sort_values("prediction", ascending=False)
    title = f"{name}: prediction vs label on {pd.Timestamp(day):%Y-%m-%d}"
    st.plotly_chart(viz.prediction_scatter(selected, title, k, dark), theme=None, width="stretch")
    st.caption(
        f"Label: the forward {cfg.pred_len}-day return as the model is scored on it (daily z-score). The {k} highest "
        "predictions are the stocks a long-only top-k portfolio would hold that day."
    )
    st.dataframe(selected.drop(columns="date"), hide_index=True)
