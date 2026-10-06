"""Plotly figures shared by the notebooks and the Streamlit app, so both draw identical charts
(THESIS_GUIDE.md Step 11).

Colors follow the dataviz reference palette: categorical slots in fixed order (validated for up to
3 series in light and dark mode), a single-hue blue ramp for magnitudes, recessive hairline axes.
`dark=True` selects the dark-mode steps; it is not an automatic inversion. Every figure is drawn
from a DataFrame that should be shown next to it as the table view.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xaifin.config import STUDY_END, STUDY_START, UNIVERSE_NAMES
from xaifin.data.features import FAMILIES

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
THEMES = {
    False: {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "band": "rgba(11,11,11,0.05)",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    True: {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "band": "rgba(255,255,255,0.06)",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}


def _style(fig: go.Figure, title: str, dark: bool, height: int = 380, right_margin: int = 24) -> go.Figure:
    t = THEMES[dark]
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=t["text"])),
        font=dict(family=FONT, size=12, color=t["text2"]),
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        height=height,
        margin=dict(l=64, r=right_margin, t=72, b=48),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom", font=dict(color=t["text2"])),
        hoverlabel=dict(font_family=FONT),
    )
    axis = dict(gridcolor=t["grid"], linecolor=t["axis"], zeroline=False,
                tickfont=dict(color=t["muted"]), title_font=dict(color=t["text2"]))
    fig.update_xaxes(showline=True, **axis)
    fig.update_yaxes(**axis)
    return fig


def _study_period(fig: go.Figure, dark: bool) -> None:
    fig.add_vrect(
        x0=STUDY_START, x1=STUDY_END, fillcolor=THEMES[dark]["band"], line_width=0, layer="below",
        annotation_text="study period", annotation_position="top left",
        annotation_font=dict(size=11, color=THEMES[dark]["muted"]),
    )


def _days(dates: pd.Series) -> pd.Series:
    """Dates as "YYYY-MM-DD": plotly would store timestamps with a time part, doubling the payload."""
    return pd.to_datetime(dates).dt.strftime("%Y-%m-%d")


def _end_labels(fig: go.Figure, ends: list[tuple], span: float, dark: bool) -> None:
    """Direct labels at the line ends, in text ink (the lines carry the color), nudged apart
    vertically so they never overlap. `ends` = [(x, y, text)], `span` = the y range of the plot."""
    gap, previous = 0.07 * span, None
    for x, y, text in sorted(ends, key=lambda end: end[1]):
        if previous is not None and y - previous < gap:
            y = previous + gap
        previous = y
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=THEMES[dark]["text2"]))


def coverage_chart(membership: pd.DataFrame, coverage: pd.DataFrame, universe: str, dark: bool = False) -> go.Figure:
    """Index members (month ends) and tickers with a price (daily) over time."""
    t = THEMES[dark]
    fig = go.Figure()
    series = [
        ("Index members", _days(membership["date"]), membership["members"], "hv"),
        ("Tickers with prices", _days(coverage["date"]), coverage["tickers"], "linear"),
    ]
    for (name, x, y, shape), color in zip(series, t["series"]):
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2, shape=shape)))
    span = max(membership["members"].max(), coverage["tickers"].max())
    _end_labels(fig, [(x.iloc[-1], y.iloc[-1], name) for name, x, y, _ in series], span, dark)
    _study_period(fig, dark)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Stocks", rangemode="tozero")
    return _style(fig, f"{UNIVERSE_NAMES[universe]}: index members and tickers with prices", dark, right_margin=140)


def missing_chart(missing: pd.DataFrame, catalog: pd.DataFrame, universe: str, dark: bool = False) -> go.Figure:
    """Mean share of missing values per feature family, Alpha158 vs Alpha360, over the study period."""
    t = THEMES[dark]
    by_family = (
        missing.merge(catalog[["group", "feature", "family"]], on=["group", "feature"])
        .groupby(["group", "family"])["missing_share"].mean().reset_index()
    )
    order = [f for f in FAMILIES if f in set(by_family["family"])]
    fig = go.Figure()
    for group, color in zip(["alpha158", "alpha360"], t["series"]):
        d = by_family[by_family["group"] == group].set_index("family").reindex(order)
        fig.add_trace(go.Bar(
            y=order, x=d["missing_share"], name=group.capitalize(), orientation="h",
            marker=dict(color=color, cornerradius=4),
            hovertemplate="%{y}: %{x:.2%} missing<extra>" + group.capitalize() + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.12)
    fig.update_xaxes(title_text="Missing values (mean share per feature)", tickformat=".1%", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, f"{UNIVERSE_NAMES[universe]}: missing values by feature family, 2015-2024", dark)


def label_chart(label_stats: pd.DataFrame, universe: str, dark: bool = False) -> go.Figure:
    """Distribution of the forward-return label per year: box = quartiles, whiskers = 5th-95th percentile."""
    t = THEMES[dark]
    pred_len = int(label_stats["pred_len"].iloc[0])
    years = label_stats["year"].astype(str)
    fig = go.Figure(go.Box(
        x=years, q1=label_stats["q1"], median=label_stats["median"], q3=label_stats["q3"],
        lowerfence=label_stats["p05"], upperfence=label_stats["p95"], mean=label_stats["mean"],
        name=f"{pred_len}-day return", marker_color=t["series"][0], line=dict(width=2),
        fillcolor="rgba(42,120,214,0.18)" if not dark else "rgba(57,135,229,0.28)", boxpoints=False,
    ))
    fig.update_yaxes(title_text=f"{pred_len}-day forward return", tickformat=".0%", hoverformat=".2%")
    fig.update_xaxes(title_text=None, type="category")
    return _style(fig, f"{UNIVERSE_NAMES[universe]}: {pred_len}-day forward return per year (whiskers: 5th-95th pct.)", dark)


def correlation_chart(corr: pd.DataFrame, clusters: pd.DataFrame, universe: str, dark: bool = False) -> go.Figure:
    """|Spearman| correlation between the Alpha158 features, in dendrogram order (clusters adjacent)."""
    t = THEMES[dark]
    order = clusters.sort_values("order")["feature"].tolist()
    z = corr.loc[order, order]
    ramp = t["ramp"]
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy().round(3), x=order, y=order, zmin=0, zmax=1,
        colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)],
        colorbar=dict(title=dict(text="|ρ|", font=dict(color=t["text2"])), tickfont=dict(color=t["muted"]), thickness=12),
        hovertemplate="%{y} vs %{x}<br>|ρ| = %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(showticklabels=False, showgrid=False, showline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed", scaleanchor="x")
    return _style(fig, f"{UNIVERSE_NAMES[universe]}: |Spearman ρ| between Alpha158 features", dark, height=620)


def volatility_chart(volatility: pd.DataFrame, universe: str, dark: bool = False) -> go.Figure:
    """Annualized 20-day realized volatility of each market index over time."""
    t = THEMES[dark]
    fig = go.Figure()
    ends = []
    for (index, d), color in zip(volatility.groupby("index", sort=False), t["series"]):
        x, y = _days(d["date"]), d["volatility"].round(4)
        fig.add_trace(go.Scatter(x=x, y=y, name=index, mode="lines", line=dict(color=color, width=2)))
        ends.append((x.iloc[-1], y.iloc[-1], index))
    _end_labels(fig, ends, volatility["volatility"].max(), dark)
    _study_period(fig, dark)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Annualized volatility", tickformat=".0%", hoverformat=".1%", rangemode="tozero")
    return _style(fig, f"{UNIVERSE_NAMES[universe]}: 20-day realized volatility of the market indexes", dark,
                  right_margin=110)


def training_chart(history: pd.DataFrame, title: str, best_epoch: int | None = None, dark: bool = False) -> go.Figure:
    """Training loss (top) and validation IC / RankIC (bottom) per epoch; `history` has epoch, train_loss, IC, RankIC."""
    t = THEMES[dark]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=history["epoch"], y=history["train_loss"].round(5), name="training loss",
                             mode="lines", line=dict(color=t["series"][0], width=2)), row=1, col=1)
    for name, color in (("IC", t["series"][1]), ("RankIC", t["series"][2])):
        if name in history:
            fig.add_trace(go.Scatter(x=history["epoch"], y=history[name].round(5), name=f"validation {name}",
                                     mode="lines", line=dict(color=color, width=2)), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1), row=2, col=1)
    if best_epoch is not None:
        fig.add_vline(x=best_epoch, line=dict(color=t["muted"], width=1, dash="dot"))
        fig.add_annotation(x=best_epoch, y=1, yref="paper", text="kept epoch", showarrow=False, xanchor="left",
                           xshift=4, yanchor="top", font=dict(size=11, color=t["muted"]))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Validation", hoverformat=".4f", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    return _style(fig, title, dark, height=480)


def daily_ic_chart(daily: pd.DataFrame, title: str, window: int = 20, dark: bool = False) -> go.Figure:
    """Daily RankIC of the test year (bars) and its `window`-day rolling mean (line); `daily` has date, RankIC."""
    t = THEMES[dark]
    x = _days(daily["date"])
    fig = go.Figure(go.Bar(x=x, y=daily["RankIC"].round(4), name="daily RankIC", marker_color=t["ramp"][2],
                           marker_line_width=0))
    fig.add_trace(go.Scatter(x=x, y=daily["RankIC"].rolling(window, min_periods=1).mean().round(4),
                             name=f"{window}-day mean", mode="lines", line=dict(color=t["series"][1], width=2)))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.update_layout(hovermode="x unified", bargap=0.1)
    fig.update_yaxes(title_text="RankIC", hoverformat=".3f")
    return _style(fig, title, dark)


def prediction_scatter(day: pd.DataFrame, title: str, k: int, dark: bool = False) -> go.Figure:
    """Prediction vs label of the stocks of one day; the k highest predictions (the top-k portfolio) are highlighted.

    `day` has ticker, prediction, label.
    """
    t = THEMES[dark]
    top = day["prediction"].rank(ascending=False, method="first") <= k
    fig = go.Figure()
    for name, mask, color, size in ((f"top {k} by prediction", top, t["series"][1], 10),
                                    ("other stocks", ~top, t["series"][0], 8)):
        d = day[mask]
        fig.add_trace(go.Scatter(x=d["prediction"].round(4), y=d["label"].round(4), name=name, mode="markers",
                                 text=d["ticker"], marker=dict(color=color, size=size, line=dict(width=0)),
                                 hovertemplate="%{text}<br>prediction %{x:.3f}<br>label %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.update_xaxes(title_text="Prediction", zeroline=False)
    fig.update_yaxes(title_text="Label (daily z-score)")
    return _style(fig, title, dark)
