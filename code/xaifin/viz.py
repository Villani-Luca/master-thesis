"""Plotly figures shared by the notebooks and the Streamlit app, so both draw identical charts
(THESIS_GUIDE.md Step 11).

Colors follow the dataviz reference palette: categorical slots in fixed order (the first 6 validated
as adjacent series in light and dark mode; slots 4-6 and two of the first three are below 3:1 on the
light surface, so charts that use them carry direct labels and a table view), a single-hue blue ramp
for magnitudes, a blue-red diverging scale with a gray midpoint for signed values, recessive axes.
`dark=True` selects the dark-mode steps; it is not an automatic inversion. Every figure is drawn
from a DataFrame that should be shown next to it as the table view.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xaifin.config import STUDY_END, STUDY_START, UNIVERSE_NAMES
from xaifin.data.features import FAMILIES

# Fixed color slot of each model: the color follows the model on every chart, whatever is filtered.
MODEL_ORDER = ["MASTER", "MATCC", "FactorVAE", "HIST", "DiscoverPLF", "FinFormer"]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
THEMES = {
    False: {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "band": "rgba(11,11,11,0.05)",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "diverging": ["#e34948", "#f0efec", "#2a78d6"],
    },
    True: {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "band": "rgba(255,255,255,0.06)",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "diverging": ["#e66767", "#383835", "#3987e5"],
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


def metric_heatmap(table: pd.DataFrame, metric: str, title: str, dark: bool = False) -> go.Figure:
    """Model x test year grid of a signed metric (IC, RankIC, ...): blue above 0, red below, gray at 0.

    `table` has model, test_year and `metric` (one row per cell, e.g. the mean over seeds).
    """
    t = THEMES[dark]
    grid = table.pivot(index="model", columns="test_year", values=metric)
    grid = grid.reindex([m for m in MODEL_ORDER if m in grid.index])
    limit = float(abs(grid).max().max()) or 1.0
    fig = go.Figure(go.Heatmap(
        z=grid.values.round(4), x=[str(y) for y in grid.columns], y=list(grid.index), zmid=0, zmin=-limit, zmax=limit,
        colorscale=[[0, t["diverging"][0]], [0.5, t["diverging"][1]], [1, t["diverging"][2]]],
        text=grid.map(lambda v: "" if pd.isna(v) else f"{v:+.3f}").values, texttemplate="%{text}",
        textfont=dict(size=12, color=t["text"]), xgap=2, ygap=2,
        colorbar=dict(title=dict(text=metric, font=dict(color=t["text2"])), tickfont=dict(color=t["muted"]),
                      outlinewidth=0, thickness=12),
        hovertemplate="%{y}, %{x}<br>" + metric + " %{z:+.4f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Test year", type="category", showline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, title, dark, height=80 + 52 * len(grid))


def metric_by_year_chart(table: pd.DataFrame, metric: str, title: str, dark: bool = False) -> go.Figure:
    """One line per model across test years; `table` has model, test_year, `metric` (optional `metric`_std band)."""
    t = THEMES[dark]
    fig = go.Figure()
    ends = []
    for model in [m for m in MODEL_ORDER if m in set(table["model"])]:
        d = table[table["model"] == model].sort_values("test_year")
        color = t["series"][MODEL_ORDER.index(model)]
        x, y = d["test_year"].astype(str), d[metric].round(4)
        fig.add_trace(go.Scatter(x=x, y=y, name=model, mode="lines+markers", line=dict(color=color, width=2),
                                 marker=dict(size=8, color=color, line=dict(color=t["surface"], width=2))))
        ends.append((x.iloc[-1], y.iloc[-1], model))
    span = float(table[metric].max() - table[metric].min()) or 1.0
    _end_labels(fig, ends, span, dark)
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Test year", type="category")
    fig.update_yaxes(title_text=metric, hoverformat="+.4f")
    return _style(fig, title, dark, right_margin=110)


BENCHMARK_STYLE = {"equal weight": "dash", "index": "dot"}


def equity_chart(curves: pd.DataFrame, title: str, dark: bool = False) -> go.Figure:
    """Growth of 1 invested, one line per column of `curves` (index = date).

    Model columns take their fixed color (MODEL_ORDER); the benchmarks 'equal weight' and 'index'
    are neutral, dashed and dotted.
    """
    t = THEMES[dark]
    fig = go.Figure()
    ends, x = [], _days(pd.Series(curves.index))
    for column in curves.columns:
        if column in MODEL_ORDER:
            line = dict(color=t["series"][MODEL_ORDER.index(column)], width=2)
        else:
            line = dict(color=t["text2"] if column == "index" else t["muted"], width=2,
                        dash=BENCHMARK_STYLE.get(column, "solid"))
        y = curves[column].round(4)
        fig.add_trace(go.Scatter(x=x, y=y, name=column, mode="lines", line=line))
        ends.append((x.iloc[-1], y.iloc[-1], column))
    _end_labels(fig, ends, float(curves.max().max() - curves.min().min()) or 1.0, dark)
    fig.add_hline(y=1, line=dict(color=t["axis"], width=1))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Value of 1 invested", hoverformat=".3f")
    return _style(fig, title, dark, right_margin=120)


def holdings_map(holdings: pd.DataFrame, title: str, dark: bool = False) -> go.Figure:
    """Which stocks the portfolio held in each period: one row per stock (most often held on top), one column per rebalance."""
    t = THEMES[dark]
    held = holdings.pivot_table(index="ticker", columns="start", values="held", aggfunc="max", fill_value=False)
    held = held.loc[held.sum(axis=1).sort_values(ascending=False).index]
    rank = holdings.pivot_table(index="ticker", columns="start", values="rank").reindex(index=held.index, columns=held.columns)
    fig = go.Figure(go.Heatmap(
        z=held.astype(int).values, x=_days(pd.Series(held.columns)), y=list(held.index), customdata=rank.values,
        colorscale=[[0, t["surface"]], [0.5, t["surface"]], [0.5, t["series"][0]], [1, t["series"][0]]], zmin=0, zmax=1,
        showscale=False, xgap=1, ygap=1,
        hovertemplate="%{y}, period from %{x}<br>rank %{customdata}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))
    fig.update_xaxes(showgrid=False)
    return _style(fig, title, dark, height=max(320, 60 + 14 * len(held)))


def margin_histogram(holdings: pd.DataFrame, title: str, dark: bool = False) -> go.Figure:
    """Distribution of the held stocks' margin to the k-boundary, in cross-sectional standard deviations of the scores."""
    t = THEMES[dark]
    margins = holdings.loc[holdings["held"], "margin_z"]
    fig = go.Figure(go.Histogram(x=margins.round(4), nbinsx=40, marker=dict(color=t["series"][0], line=dict(width=0)),
                                 name="held stocks", hovertemplate="margin %{x}<br>%{y} holdings<extra></extra>"))
    fig.update_layout(bargap=0.08)
    fig.update_xaxes(title_text="Margin to the k-boundary (score standard deviations of the day)")
    fig.update_yaxes(title_text="Holdings (stock x period)")
    return _style(fig, title, dark)


def overlap_heatmap(matrix: pd.DataFrame, title: str, dark: bool = False) -> go.Figure:
    """Average overlap (0-1) of the stocks held by pairs of portfolios: one-hue scale, value in each cell."""
    t = THEMES[dark]
    steps = t["ramp"] if not dark else t["ramp"][::-1]
    fig = go.Figure(go.Heatmap(
        z=matrix.values.round(3), x=list(matrix.columns), y=list(matrix.index), zmin=0, zmax=1,
        colorscale=[[i / (len(steps) - 1), c] for i, c in enumerate(steps)],
        text=matrix.map(lambda v: "" if pd.isna(v) else f"{v:.2f}").values, texttemplate="%{text}",
        textfont=dict(size=12), xgap=2, ygap=2,
        colorbar=dict(title=dict(text="overlap", font=dict(color=t["text2"])), tickfont=dict(color=t["muted"]),
                      outlinewidth=0, thickness=12),
        hovertemplate="%{y} vs %{x}<br>overlap %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return _style(fig, title, dark, height=80 + 52 * len(matrix))
