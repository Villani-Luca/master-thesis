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
Chapter 6 (performance, from the results store, notebook 03):
- res_rankic: mean test RankIC per model and test year, one heatmap per universe (L = 5);
- res_horizons: mean test RankIC per model at the three horizons, with standard errors.
Chapter 6 (explanations, from results/xai/, IG of the score, core universes, L = 5, notebooks 07 and 08):
- res_inputs (table): net family shares and the share of the most recent and oldest days, per model;
- res_agreement (table): agreement of two runs that differ in one factor, families and features;
- res_agreement_models (table): per model, RankIC and feature agreement with another seed, model and test year;
- res_variance (table): variance decomposition of the net family shares, per feature group.
Chapter 7 (portfolios, notebooks 04 and 09):
- pf_performance, pf_fragility (tables): CAGR and Sharpe per model and universe; fragile holdings against seed turnover;
- pf_equity: growth of 1 invested, seed 42, L = 5, DJI and NASDAQ-100;
- pf_fragility_scatter: share of fragile holdings against the turnover observed between seeds.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from xaifin import viz
from xaifin.config import CODE_ROOT, CORE_UNIVERSES, RESULTS_ROOT, STUDY_END, STUDY_START, TOP_K, UNIVERSE_NAMES, UNIVERSES
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


def _table(name: str, columns: str, header: list[str], body: list[str], caption: str) -> Path:
    tex = "\n".join(["\\begin{table}[htbp]", "\\centering", "\\small", "\\begin{adjustbox}{max width=\\textwidth}",
                     f"\\begin{{tabular}}{{{columns}}}", "\\toprule", *header, "\\midrule", *body, "\\bottomrule",
                     "\\end{tabular}", "\\end{adjustbox}", f"\\caption{{{caption}}}", f"\\label{{tab:{name}}}", "\\end{table}", ""])
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"{name}.tex"
    out.write_text(tex, encoding="utf-8")
    return out


def pf_performance() -> Path:
    """LaTeX table: CAGR and Sharpe of the seed-42 portfolios at L = 5 per model and universe, with the benchmarks
    (results/analysis/portfolio_summary.csv, notebook 04)."""
    s = pd.read_csv(RESULTS_ROOT / "analysis" / "portfolio_summary.csv")
    rows = [*viz.MODEL_ORDER, "equal weight", "index"]
    names = {u: UNIVERSE_NAMES[u].replace("&", "\\&") for u in UNIVERSES}
    header = ["\\textbf{Portfolio} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{names[u]}}}}}" for u in UNIVERSES) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(UNIVERSES))),
              " & " + " & ".join("CAGR & Sharpe" for _ in UNIVERSES) + " \\\\"]
    body = []
    for row in rows:
        if row == "equal weight":
            body.append("\\midrule")
        cells = []
        for u in UNIVERSES:
            r = s[(s["universe"] == u) & (s["portfolio"] == row)]
            ew = s[(s["universe"] == u) & (s["portfolio"] == "equal weight")].iloc[0]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            bold = row in viz.MODEL_ORDER and r["CAGR"] > ew["CAGR"]
            cagr = f"{100 * r['CAGR']:+.1f}"
            cells += [f"\\textbf{{{cagr}}}" if bold else cagr, f"{r['Sharpe']:.2f}"]
        body.append(f"{row} & " + " & ".join(cells) + " \\\\")
    return _table("pf_performance", "@{}l" + "rr" * len(UNIVERSES) + "@{}", header, body,
                  "Long-only top-$k$ portfolios at $L = 5$ (seed 42), 2020--2024 (2021--2024 for the European universes): "
                  "compound annual growth (\\%) and Sharpe ratio of each model's portfolio, of the equal-weight portfolio of all "
                  "the stocks the model ranks, and of the index. In bold, the CAGRs above the equal-weight benchmark.")


def pf_fragility() -> Path:
    """LaTeX table: share of held positions closer to the boundary than their seed noise, against the turnover observed
    between seeds (results/analysis/decision_fragility.csv, notebook 09)."""
    f = pd.read_csv(RESULTS_ROOT / "analysis" / "decision_fragility.csv")
    universes = [u for u in CORE_UNIVERSES if u in set(f["universe"])]
    header = ["\\textbf{Model} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{UNIVERSE_NAMES[u]}}}}}" for u in universes) + " \\\\",
              " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(universes))),
              " & " + " & ".join("fragile & changed" for _ in universes) + " \\\\"]
    body = []
    for model in viz.MODEL_ORDER:
        cells = []
        for u in universes:
            r = f[(f["model"] == model) & (f["universe"] == u)].iloc[0]
            cells += [f"{100 * r['share_below_noise']:.0f}", f"{100 * r['observed seed turnover']:.0f}"]
        body.append(f"{model} & " + " & ".join(cells) + " \\\\")
    return _table("pf_fragility", "@{}l" + "rr" * len(universes) + "@{}", header, body,
                  "Fragility of the allocation decisions at $L = 5$. \\emph{Fragile}: share (\\%) of the held positions whose "
                  "margin to the top-$k$ boundary is smaller than the seed noise of the stock's score on that day (standard "
                  "deviation of its daily z-score across the three seeds). \\emph{Changed}: share (\\%) of the portfolio that "
                  "differs between two seeds, averaged over pairs of seeds and rebalances.")


def _metrics() -> pd.DataFrame:
    from xaifin.training.rolling import collect_results

    return collect_results()


def _importance(level: str, value: str = "importance", target: str = "pred") -> pd.DataFrame:
    """IG global importance of the core runs at L = 5 (notebook 07), without the market inputs (zero by construction)."""
    from xaifin.analysis.variance import load_importance
    from xaifin.data.features import market_names

    t = load_importance("ig_joint", target, level, value)
    t = t[(t["pred_len"] == 5) & t["universe"].isin(CORE_UNIVERSES)]
    market = {(u, n) for u in CORE_UNIVERSES for n in [*market_names(u), "market"]}
    return t[[(u, n) not in market for u, n in zip(t["universe"], t["name"])]]


def _model_pairs(pairs: pd.DataFrame) -> pd.Series:
    """Mean Spearman of each model with the other models (pairs counted for both)."""
    p = pairs[pairs["factor"] == "model"]
    return pd.concat([p.assign(m=p["a"]), p.assign(m=p["b"])]).groupby("m")["spearman"].mean()


FACTOR_NAMES = {"seed": "another seed", "universe": "another universe", "year": "another test year",
                "model": "another model of the group"}


def res_inputs() -> Path:
    """LaTeX table: per model, the net IG share of each family and the gross share of the most recent and oldest days
    (notebooks 05 and 07)."""
    family = _importance("family", "net")
    family["share"] = family["importance"] / family.groupby(["model", "universe", "seed", "test_year"])["importance"].transform("sum")
    families = family.pivot_table(index="name", columns="model", values="share", aggfunc="mean")
    lag = _importance("lag")
    lag["lag"] = lag["name"].astype(int)
    lag["oldest"] = lag.groupby("model")["lag"].transform("max")
    runs = ["model", "universe", "seed", "test_year"]
    lags = pd.DataFrame({"most recent day": lag[lag["lag"] == 0].groupby(runs)["importance"].sum(),
                         "most recent 5 days": lag[lag["lag"] < 5].groupby(runs)["importance"].sum(),
                         "oldest 5 days": lag[lag["lag"] > lag["oldest"] - 5].groupby(runs)["importance"].sum()}).groupby("model").mean().T
    models = viz.MODEL_ORDER
    header = ["& " + " & ".join(f"\\textbf{{{m}}}" for m in models) + " \\\\"]
    body = ["\\multicolumn{7}{@{}l}{\\emph{Families (net share)}} \\\\"]
    order = families[models[0]].sort_values(ascending=False).index
    for name in order:
        cells = ["--" if pd.isna(families.loc[name, m]) else f"{100 * families.loc[name, m]:.0f}" for m in models]
        body.append(f"{name.replace('_', '-')} & " + " & ".join(cells) + " \\\\")
    body += ["\\midrule", "\\multicolumn{7}{@{}l}{\\emph{Days of the window (gross share)}} \\\\"]
    for name, row in lags.iterrows():
        body.append(f"{name} & " + " & ".join(f"{100 * row[m]:.0f}" for m in models) + " \\\\")
    return _table("res_inputs", "@{}l" + "r" * len(models) + "@{}", header, body,
                  "What the models rely on: IG of the score, core universes, $L = 5$, mean over the 42 runs of each model "
                  "(\\%). Families: net share of each family (market inputs left out); the Alpha360 set has three families. "
                  "Days: gross share of the attribution received by the most recent day, the most recent five days and the "
                  "oldest five days of the window, which is 20 days for the Alpha158 models and 60 for the Alpha360 models "
                  "(uniform: 5\\%, 25\\% and 25\\%, and 1.7\\%, 8.3\\% and 8.3\\%).")


def res_agreement() -> Path:
    """LaTeX table: agreement of the IG importance rankings of two runs that differ in one factor only (notebook 07)."""
    from xaifin.analysis.variance import pair_agreement

    family = _importance("family", "net")
    family = pair_agreement(family[family["group"] == "alpha158"]).groupby("factor")["spearman"].mean()
    p = pair_agreement(_importance("feature"))
    feature = p.groupby("factor").agg(spearman=("spearman", "mean"), jaccard=("jaccard", "mean"), pairs=("spearman", "size"))
    header = ["\\textbf{The two runs differ in} & \\textbf{Families (Alpha158)} & \\multicolumn{2}{c}{\\textbf{Features}} & "
              "\\textbf{Pairs} \\\\", "\\cmidrule(lr){2-2} \\cmidrule(lr){3-4}", " & Spearman & Spearman & top-10 Jaccard & \\\\"]
    body = [f"{FACTOR_NAMES[f]} & {family[f]:.2f} & {feature.loc[f, 'spearman']:.2f} & {feature.loc[f, 'jaccard']:.2f} & "
            f"{feature.loc[f, 'pairs']} \\\\" for f in FACTOR_NAMES]
    return _table("res_agreement", "@{}lrrrr@{}", header, body,
                  "Agreement of the explanations of two runs that differ in one factor only, IG of the score, core universes, "
                  "$L = 5$, mean over all such pairs. Families: net shares of the seven Alpha158 families (the Alpha360 "
                  "models, with three families, always agree). Features: gross importance, within a feature group. "
                  "The market inputs, whose attribution is zero by construction, are left out.")


def res_agreement_models() -> Path:
    """LaTeX table: per model, its skill and the feature-level agreement of its explanations with its other seeds, the
    other models of its group and its next test year (notebooks 07 and 08)."""
    from xaifin.analysis.variance import pair_agreement

    p = pair_agreement(_importance("feature"))
    seed = p[p["factor"] == "seed"].groupby("model")["spearman"].mean()
    years = p[p["factor"] == "year"].astype({"a": int, "b": int})
    next_year = years[(years["b"] - years["a"]).abs() == 1].groupby("model")["spearman"].mean()
    metrics = _metrics()
    rankic = metrics[(metrics["pred_len"] == 5) & metrics["universe"].isin(CORE_UNIVERSES)].groupby("model")["RankIC"].mean()
    others = _model_pairs(p)
    header = ["\\textbf{Model} & \\textbf{Mean RankIC} & \\multicolumn{3}{c}{\\textbf{Feature agreement (Spearman) with}} \\\\",
              "\\cmidrule(lr){3-5}", " & & another seed & another model & the next test year \\\\"]
    body = []
    for model in viz.MODEL_ORDER:
        if model == "HIST":
            body.append("\\midrule")
        body.append(f"{model} & {rankic[model]:+.3f} & {seed[model]:.2f} & {others[model]:.2f} & {next_year[model]:.2f} \\\\")
    return _table("res_agreement_models", "@{}lrrrr@{}", header, body,
                  "Skill and stability of each model's explanations, core universes, $L = 5$. Mean test RankIC of the 42 "
                  "runs; mean Spearman correlation between the feature importances (IG of the score, market inputs left out) "
                  "of the run and of the same run with another seed, of another model of its feature group (same seed, "
                  "universe and year), and of the next test year with the same seed. Above the line the Alpha158 group, "
                  "below it the Alpha360 group.")


def res_variance() -> Path:
    """LaTeX table: share of the variance of each family's net IG share across runs explained by each factor (notebook 07)."""
    from xaifin.analysis.variance import decomposition

    family = _importance("family", "net")
    parts = ["model", "universe", "year", "interactions", "seed"]
    header = ["\\textbf{Family} & \\textbf{Mean share} & " + " & ".join(f"\\textbf{{{c.capitalize()}}}" for c in parts) + " \\\\"]
    body = []
    for group, title in (("alpha158", "Alpha158 group (MASTER, MATCC, FactorVAE)"), ("alpha360", "Alpha360 group (HIST, DiscoverPLF, FinFormer)")):
        d = decomposition(family[family["group"] == group]).set_index("name")
        body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for name, r in d.iterrows():
            label = "all, weighted" if name.startswith("all") else name.replace("_", "-")
            share = "" if pd.isna(r["mean_importance"]) else f"{100 * r['mean_importance']:.0f}"
            cells = " & ".join(f"{100 * r[c]:.0f}" for c in parts)
            body.append(("\\midrule " if name.startswith("all") else "") + f"{label} & {share} & {cells} \\\\")
        if group == "alpha158":
            body.append("\\midrule")
    return _table("res_variance", "@{}lrrrrrr@{}", header, body,
                  "Where the differences between explanations come from: share (\\%) of the variance of each family's net IG "
                  "share across the 126 runs of a feature group explained by the model, the universe and the test year "
                  "(type II sums of squares), their interactions, and the seed (spread of the three seeds within each "
                  "model, universe and year), core universes, $L = 5$. The last row of each group averages the families "
                  "weighted by their mean share.")


def res_rankic() -> Path:
    """Model x test year heatmaps of the mean RankIC over seeds (L = 5), one per universe, on one shared scale."""
    runs = _metrics()
    runs = runs[runs["pred_len"] == 5]
    cells = runs.groupby(["universe", "model", "test_year"])["RankIC"].mean().reset_index()
    limit = float(cells["RankIC"].abs().max())
    order = [*CORE_UNIVERSES, *[u for u in UNIVERSES if u not in CORE_UNIVERSES]]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[UNIVERSE_NAMES[u] for u in order], horizontal_spacing=0.04,
                        vertical_spacing=0.14, shared_yaxes=True)
    for i, universe in enumerate(order):
        heat = viz.metric_heatmap(cells[cells["universe"] == universe], "RankIC", "").data[0]
        heat.update(zmin=-limit, zmax=limit, showscale=i == 0, textfont=dict(size=10, color=LIGHT["text"]),
                    colorbar=dict(title=dict(text="RankIC"), thickness=10, x=1.0, len=0.9))
        fig.add_trace(heat, row=i // 3 + 1, col=i % 3 + 1)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(type="category", showline=False)
    viz._style(fig, "", False)
    fig.update_annotations(font=dict(size=13, color=LIGHT["text"]))
    return save(fig, "res_rankic", 560)


def res_horizons() -> Path:
    """Mean RankIC per model at L = 1, 5, 20 (over universes, years and seeds), error bars = standard error of the mean.
    The horizon is ordinal, so it takes three steps of the sequential ramp rather than categorical hues."""
    runs = _metrics()
    stats = runs.groupby(["pred_len", "model"])["RankIC"].agg(["mean", "sem"]).reset_index()
    ramp = LIGHT["ramp"]
    fig = go.Figure()
    for pred_len, color in zip((1, 5, 20), (ramp[2], ramp[4], ramp[6])):
        d = stats[stats["pred_len"] == pred_len].set_index("model").reindex(viz.MODEL_ORDER)
        fig.add_trace(go.Bar(x=viz.MODEL_ORDER, y=d["mean"], name=f"L = {pred_len}", marker_color=color,
                             error_y=dict(type="data", array=d["sem"], color=LIGHT["text2"], thickness=1.2, width=4)))
    fig.add_hline(y=0, line=dict(color=LIGHT["axis"], width=1))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title_text="Mean test RankIC", tickformat="+.3f")
    viz._style(fig, "", False)
    return save(fig, "res_horizons", 380)


def pf_equity() -> Path:
    """Growth of 1 invested in the seed-42 portfolios at L = 5 and in the two benchmarks, DJI and NASDAQ-100."""
    import json

    from xaifin.portfolio.backtest import find_portfolios

    portfolios = find_portfolios()
    portfolios = portfolios[(portfolios["pred_len"] == 5) & (portfolios["seed"] == 42) & (portfolios["k"] == portfolios["universe"].map(TOP_K))]
    panels = ["dji", "nasdaq100"]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, subplot_titles=[UNIVERSE_NAMES[u] for u in panels])
    fig.update_annotations(font=dict(size=13, color=LIGHT["text"]), x=0, xanchor="left")
    for row, universe in enumerate(panels, 1):
        curves = {}
        for r in portfolios[portfolios["universe"] == universe].itertuples():
            ret = pd.read_parquet(r.path / "returns.parquet").set_index("date")
            curves[r.model] = (1 + ret["portfolio"]).cumprod()
            bench = ret
        curves = pd.DataFrame({m: curves[m] for m in viz.MODEL_ORDER if m in curves})
        curves["equal weight"], curves["index"] = (1 + bench["equal_weight"]).cumprod(), (1 + bench["index"]).cumprod()
        for trace in viz.equity_chart(curves, "").data:
            fig.add_trace(trace.update(showlegend=row == 1), row=row, col=1)
        fig.add_hline(y=1, line=dict(color=LIGHT["axis"], width=1), row=row, col=1)
        fig.update_yaxes(title_text="Value of 1", row=row, col=1)
    viz._style(fig, "", False)
    fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=-0.08, yanchor="top"), margin=dict(b=80))
    return save(fig, "pf_equity", 700)


def pf_fragility_scatter() -> Path:
    """Share of holdings within their seed noise of the boundary against the share of the portfolio that changes
    between seeds; model = fixed color slot, universe = marker symbol; the dotted diagonal is equality."""
    f = pd.read_csv(RESULTS_ROOT / "analysis" / "decision_fragility.csv")
    symbols = dict(zip(CORE_UNIVERSES, ("circle", "square", "diamond")))
    fig = go.Figure()
    for i, model in enumerate(viz.MODEL_ORDER):
        d = f[f["model"] == model]
        fig.add_trace(go.Scatter(x=d["share_below_noise"], y=d["observed seed turnover"], mode="markers", name=model,
                                 marker=dict(color=LIGHT["series"][i], size=11, symbol=[symbols[u] for u in d["universe"]],
                                             line=dict(color="white", width=2))))
    for universe, symbol in symbols.items():  # legend entries for the symbols, in neutral ink
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=UNIVERSE_NAMES[universe],
                                 marker=dict(color=LIGHT["muted"], size=10, symbol=symbol)))
    fig.add_shape(type="line", x0=0.35, y0=0.35, x1=0.8, y1=0.8, line=dict(color=LIGHT["muted"], dash="dot", width=1))
    fig.update_xaxes(title_text="Held positions closer to the boundary than their seed noise", tickformat=".0%", range=[0.35, 0.8])
    fig.update_yaxes(title_text="Portfolio changed between two seeds", tickformat=".0%", range=[0.35, 0.8])
    viz._style(fig, "", False)
    fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top"))
    return save(fig, "pf_fragility_scatter", 560)


FIGURES = {f.__name__: f for f in (data_labels, data_label_std, data_alpha158_corr, data_volatility, pf_performance,
                                   pf_fragility, res_rankic, res_horizons, res_inputs, res_agreement, res_agreement_models, res_variance,
                                   pf_equity, pf_fragility_scatter)}


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
