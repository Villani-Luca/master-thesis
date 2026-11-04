"""Figures and tables of the thesis, exported to latex-thesis/images/results/ and latex-thesis/tables/ (fixed names).

    python code/scripts/thesis_figures.py                 # every figure and table below
    python code/scripts/thesis_figures.py data_labels     # only some of them

The charts are those of xaifin.viz (the notebooks and the app draw the same ones), in the light theme,
without their title (the LaTeX caption carries it), exported to PDF with kaleido. Chapter 3 (data, from
the Step 1 summaries in results/data_summary/, notebook 01):
- data_labels: distribution of the 5-day label per year after the data-quality filter, core universes;
- data_label_std (table): standard deviation of the label per year, before and after the filter;
- data_alpha158_corr: |Spearman rho| between the Alpha158 features in cluster order, NASDAQ-100;
- data_volatility: 20-day realized volatility of the market indexes, US and Europe.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xaifin import viz
from xaifin.config import CODE_ROOT, CORE_UNIVERSES, STUDY_END, STUDY_START, UNIVERSE_NAMES, UNIVERSES
from xaifin.data.summary import summarize

THESIS = CODE_ROOT.parent / "latex-thesis"
IMAGES, TABLES = THESIS / "images" / "results", THESIS / "tables"
LIGHT = viz.THEMES[False]
WIDTH = 900  # px at scale 1; the PDF is vector, so only the proportions and the font size matter
INDEX_NAMES = {"DJI.INDX": "DJI", "NDX.INDX": "NASDAQ-100", "GSPC.INDX": "S&P 500", "SX5E.INDX": "EURO STOXX 50",
               "SXXP.INDX": "STOXX Europe 600"}


def save(fig: go.Figure, name: str, height: int) -> Path:
    fig.update_layout(title=None, margin=dict(t=40), width=WIDTH, height=height, paper_bgcolor="white", plot_bgcolor="white")
    IMAGES.mkdir(parents=True, exist_ok=True)
    out = IMAGES / f"{name}.pdf"
    fig.write_image(out)
    return out


def data_labels() -> Path:
    """Box per year (quartiles, whiskers 5th-95th percentile) of the 5-day label after the filter, one row per core universe."""
    fig = make_subplots(rows=len(CORE_UNIVERSES), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[UNIVERSE_NAMES[u] for u in CORE_UNIVERSES])
    for row, universe in enumerate(CORE_UNIVERSES, 1):
        trace = viz.label_chart(summarize(universe)["label_stats_clean"], universe).data[0]
        fig.add_trace(trace.update(showlegend=False), row=row, col=1)
        fig.update_yaxes(tickformat=".0%", title_text="5-day return" if row == 2 else None, row=row, col=1)
    fig.update_xaxes(type="category")
    viz._style(fig, "", False)
    fig.update_annotations(font=dict(size=13, color=LIGHT["text"]), x=0, xanchor="left")
    return save(fig, "data_labels", 640)


def data_label_std() -> Path:
    """LaTeX table: standard deviation of the 5-day label per year and universe, before and after the filter."""
    rows = []
    for universe in UNIVERSES:
        s = summarize(universe)
        for version, table in (("before", "label_stats"), ("after", "label_stats_clean")):
            rows.append(pd.Series(s[table].set_index("year")["std"], name=(UNIVERSE_NAMES[universe], version)))
    table = pd.DataFrame(rows)
    years = [int(y) for y in table.columns]
    cell = lambda v: "--" if pd.isna(v) else f"{100 * v:.1f}"
    body = []
    for (universe, version), values in table.iterrows():
        first = f"\\multirow{{2}}{{*}}{{{universe.replace('&', '\\&')}}}" if version == "before" else ""
        body.append(f"{first} & {version} & " + " & ".join(cell(v) for v in values) + " \\\\")
        if version == "after" and universe != table.index[-1][0]:
            body.append("\\midrule")
    tex = "\n".join([
        "\\begin{table}[htbp]", "\\centering", "\\small", "\\begin{adjustbox}{max width=\\textwidth}",
        f"\\begin{{tabular}}{{@{{}}ll{'r' * len(years)}@{{}}}}", "\\toprule",
        "\\textbf{Universe} & \\textbf{Filter} & " + " & ".join(f"\\textbf{{{y}}}" for y in years) + " \\\\",
        "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}",
        "\\caption{Standard deviation (\\%) of the five-day forward return of each calendar year, over all stock-days "
        "of the universe, before and after the data-quality filter of Section~\\ref{sec:data_quality}.}",
        "\\label{tab:data_label_std}", "\\end{table}", ""])
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "data_label_std.tex"
    out.write_text(tex, encoding="utf-8")
    return out


def data_alpha158_corr(universe: str = "nasdaq100") -> Path:
    s = summarize(universe)
    fig = viz.correlation_chart(s["alpha158_corr"], s["alpha158_clusters"], universe)
    fig.update_layout(margin=dict(l=24, r=24, b=24))
    fig.update_xaxes(showline=False)
    fig.update_yaxes(showline=False)
    return save(fig, "data_alpha158_corr", 780)


def data_volatility() -> Path:
    """Two panels (US, Europe) over the study period, test years shaded; each index keeps its own color slot
    across both panels and has a direct label at its end."""
    panels = {"US indexes": "dji", "European indexes": "sx5e"}
    slot = {index: i for i, index in enumerate(INDEX_NAMES)}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=list(panels))
    fig.update_annotations(font=dict(size=13, color=LIGHT["text"]), x=0, xanchor="left")
    for row, universe in enumerate(panels.values(), 1):
        vol = summarize(universe)["market_volatility"]
        vol = vol[(vol["date"] >= STUDY_START) & (vol["date"] <= STUDY_END)]
        ends = []
        for index, d in vol.groupby("index", sort=False):
            x, y = viz._days(d["date"]), d["volatility"]
            fig.add_trace(go.Scatter(x=x, y=y, name=INDEX_NAMES[index], mode="lines",
                                     line=dict(color=LIGHT["series"][slot[index]], width=1.5)), row=row, col=1)
            ends.append((x.iloc[-1], y.iloc[-1], INDEX_NAMES[index]))
        gap, previous = 0.08 * vol["volatility"].max(), None
        for x, y, text in sorted(ends, key=lambda end: end[1]):  # nudged apart, as viz._end_labels
            y = y if previous is None or y - previous >= gap else previous + gap
            previous = y
            fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=6,
                               font=dict(size=11, color=LIGHT["text2"]), row=row, col=1)
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title_text="Volatility", row=row, col=1)
    fig.add_vrect(x0="2020-01-01", x1=STUDY_END, fillcolor=LIGHT["band"], line_width=0, layer="below", row="all", col=1)
    fig.add_annotation(x="2020-01-01", y=1.0, xref="x", yref="paper", text="test years", showarrow=False, xanchor="left",
                       yanchor="bottom", font=dict(size=11, color=LIGHT["muted"]))
    viz._style(fig, "", False, right_margin=130)
    fig.update_xaxes(range=[STUDY_START, STUDY_END])
    fig.update_layout(showlegend=False)  # every line has its direct label
    return save(fig, "data_volatility", 620)


FIGURES = {f.__name__: f for f in (data_labels, data_label_std, data_alpha158_corr, data_volatility)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"some of {', '.join(FIGURES)} (default: all)")
    names = parser.parse_args().names or list(FIGURES)
    unknown = set(names) - set(FIGURES)
    if unknown:
        parser.error(f"unknown: {', '.join(sorted(unknown))}")
    for name in names:
        print(f"{name}: {FIGURES[name]()}")


if __name__ == "__main__":
    main()
