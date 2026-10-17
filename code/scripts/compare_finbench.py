"""Side-by-side comparison of our runs with FinBench's published results (paper Tables 3 and 7).

    python code/scripts/compare_finbench.py

Regression: test MSE and MAE from results/analysis/metrics_summary.csv against finbench_table2.csv.
As FinBench: each run's metric is averaged over the test years, then over the seeds; we also report
the standard deviation between seeds (of the per-seed averages over the years).

Portfolio: CAGR of the long-only top-5 portfolios (FinBench's k for DJI, SX5E and NASDAQ100, Table 7)
from results/portfolio/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/top5/metrics.json, averaged over the
seeds, against finbench_table7.csv. Produce them first with
    python code/scripts/run_portfolio.py --k 5

Writes results/analysis/finbench_comparison_{regression,portfolio}.csv and the thesis tables
latex-thesis/tables/finbench_{regression,portfolio}.tex.
"""

import json
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = CODE_ROOT / "results" / "analysis"
PORTFOLIO_ROOT = CODE_ROOT / "results" / "portfolio"
TABLES = CODE_ROOT.parent / "latex-thesis" / "tables"

UNIVERSE_NAMES = {"dji": "DJI", "nasdaq100": "NASDAQ 100", "sx5e": "SX5E", "sp500": "S\\&P 500", "sxxp": "SXXP"}
MODEL_ORDER = ["MASTER", "MATCC", "FactorVAE", "HIST", "DiscoverPLF", "FinFormer"]
PORTFOLIO_K = 5


def years_label(years) -> str:
    years = sorted(int(y) for y in years)
    return f"{years[0]}-{years[-1]}"


def regression() -> pd.DataFrame:
    ours = pd.read_csv(ANALYSIS / "metrics_summary.csv")
    ours = ours[ours["clean"]]
    keys = ["model", "universe", "seq_len", "pred_len"]
    per_seed = ours.groupby(keys + ["seed"]).agg(MSE=("MSE", "mean"), MAE=("MAE", "mean"),
                                                  years=("test_year", years_label)).reset_index()
    table = per_seed.groupby(keys).agg(MSE=("MSE", "mean"), MSE_sd=("MSE", "std"), MAE=("MAE", "mean"),
                                       MAE_sd=("MAE", "std"), seeds=("seed", "size"),
                                       test_years=("years", "first")).reset_index()
    finbench = pd.read_csv(ANALYSIS / "finbench_table2.csv").rename(columns={"MSE": "FB_MSE", "MAE": "FB_MAE"})
    table = table.merge(finbench, on=["model", "universe", "pred_len"], how="inner")
    table["FB_test_years"] = "2020-2024"
    for metric in ("MSE", "MAE"):
        table[f"delta_{metric}_pct"] = 100 * (table[metric] / table[f"FB_{metric}"] - 1)
    return table


def portfolio() -> pd.DataFrame:
    rows = []
    for metrics_file in sorted(PORTFOLIO_ROOT.glob(f"*/*/*/*/top{PORTFOLIO_K}/metrics.json")):
        metrics = json.loads(metrics_file.read_text())
        setup = metrics["setup"]
        rows.append({**{k: setup[k] for k in ("model", "universe", "seq_len", "pred_len", "seed")},
                     "test_years": years_label(setup["test_years"]),
                     "CAGR": metrics["portfolio"]["CAGR"], "index_CAGR": metrics["index"]["CAGR"]})
    if not rows:
        return pd.DataFrame()
    keys = ["model", "universe", "seq_len", "pred_len"]
    table = pd.DataFrame(rows).groupby(keys).agg(CAGR=("CAGR", "mean"), CAGR_sd=("CAGR", "std"),
                                                 seeds=("seed", "size"), index_CAGR=("index_CAGR", "mean"),
                                                 test_years=("test_years", "first")).reset_index()
    finbench = pd.read_csv(ANALYSIS / "finbench_table7.csv")
    benchmark = finbench[finbench["model"] == "Benchmark"].drop(columns="model").rename(columns={"CAGR": "FB_index_CAGR"})
    finbench = finbench[finbench["model"] != "Benchmark"].rename(columns={"CAGR": "FB_CAGR"})
    table = table.merge(finbench, on=["model", "universe", "pred_len"], how="left")
    table = table.merge(benchmark, on=["universe", "pred_len"], how="left")
    table["FB_test_years"] = "2020-2024"
    table["k"] = PORTFOLIO_K
    return table


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    uni = {u: i for i, u in enumerate(UNIVERSE_NAMES)}
    return table.sort_values(["pred_len", "universe", "model"],
                             key=lambda s: s.map(uni) if s.name == "universe" else s.map(rank) if s.name == "model" else s)


def num(value: float, digits: int = 3) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def lower_bold(ours: float, theirs: float, digits: int = 3) -> tuple[str, str]:
    """Format both values, the lower one (the better error) in bold."""
    a, b = num(ours, digits), num(theirs, digits)
    if pd.isna(ours) or pd.isna(theirs) or a == b:
        return a, b
    return (f"\\textbf{{{a}}}", b) if ours < theirs else (a, f"\\textbf{{{b}}}")


def table_tex(body: list[str], columns: str, header: list[str], caption: str, label: str) -> str:
    return "\n".join([
        "\\begin{table}[tb]", "\\centering", f"\\caption{{{caption}}}",
        "\\begin{adjustbox}{max width=\\textwidth}", "{\\normalsize", "\\setlength{\\tabcolsep}{6pt}", "",
        f"\\begin{{tabular}}{{{columns}}}", "\\toprule", *header, "\\midrule", *body, "\\bottomrule",
        "\\end{tabular}", "}", "\\end{adjustbox}", f"\\label{{{label}}}", "\\end{table}", ""])


def regression_tex(table: pd.DataFrame) -> str:
    body = []
    for (pred_len, universe), g in table.groupby(["pred_len", "universe"], sort=False):
        if body:
            body.append("\\midrule")
        years = g["test_years"].iloc[0]
        name = UNIVERSE_NAMES[universe] + ("" if years == "2020-2024" else f"$^\\dagger$")
        for i, r in enumerate(g.itertuples()):
            mse, fb_mse = lower_bold(r.MSE, r.FB_MSE)
            mae, fb_mae = lower_bold(r.MAE, r.FB_MAE)
            first = f"\\multirow{{{len(g)}}}{{*}}{{{name}}}" if i == 0 else ""
            body.append(f"{first} & {r.model} & {mse} {{\\scriptsize$\\pm${num(r.MSE_sd)}}} & {fb_mse} & "
                        f"{r.delta_MSE_pct:+.1f}\\% & {mae} {{\\scriptsize$\\pm${num(r.MAE_sd)}}} & {fb_mae} & "
                        f"{r.delta_MAE_pct:+.1f}\\% \\\\")
    pred_lens = sorted(table["pred_len"].unique())
    header = ["\\multirow{2}{*}{\\textbf{Universe}} & \\multirow{2}{*}{\\textbf{Model}} & "
              "\\multicolumn{3}{c}{\\textbf{MSE}} & \\multicolumn{3}{c}{\\textbf{MAE}} \\\\",
              "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
              " & & Ours & FinBench & $\\Delta$ & Ours & FinBench & $\\Delta$ \\\\"]
    caption = ("Regression error of our runs and of FinBench~\\cite{pederzoli2026finbench} "
               f"(horizon $L={', '.join(map(str, pred_lens))}$, lookback $T=20$), "
               "averaged over the test years and the seeds 0, 5, 42; $\\pm$: standard deviation between seeds. "
               "$\\Delta$: our error relative to FinBench's; the lower error is in bold. "
               "$^\\dagger$Our SX5E test years are 2021--2024 (constituent history starts in September 2020), "
               "FinBench's 2020--2024.")
    return table_tex(body, "llrrrrrr", header, caption, "tab:finbench_regression")


def portfolio_tex(table: pd.DataFrame) -> str:
    body = []
    for (pred_len, universe), g in table.groupby(["pred_len", "universe"], sort=False):
        if body:
            body.append("\\midrule")
        years = g["test_years"].iloc[0]
        name = UNIVERSE_NAMES[universe] + ("" if years == "2020-2024" else f"$^\\dagger$")
        body.append(f"\\multirow{{{len(g) + 1}}}{{*}}{{{name}}} & Index & {num(g['index_CAGR'].iloc[0], 2)} & "
                    f"{num(g['FB_index_CAGR'].iloc[0], 2)} \\\\")
        for r in g.itertuples():
            body.append(f" & {r.model} & {num(r.CAGR, 2)} {{\\scriptsize$\\pm${num(r.CAGR_sd, 2)}}} & {num(r.FB_CAGR, 2)} \\\\")
    header = ["\\textbf{Universe} & \\textbf{Model} & \\textbf{Ours} & \\textbf{FinBench} \\\\"]
    caption = (f"CAGR of the long-only top-{PORTFOLIO_K} portfolios (equal weights, rebalanced every $L=5$ "
               "trading days) of our runs and of FinBench~\\cite{pederzoli2026finbench} (Table 7), as a fraction; ours averaged "
               "over the seeds 0, 5, 42 ($\\pm$: standard deviation between seeds). Index: CAGR of the universe's "
               "index over the same days. --: not reported by FinBench. "
               "$^\\dagger$Our SX5E test years are 2021--2024, FinBench's 2020--2024.")
    return table_tex(body, "llrr", header, caption, "tab:finbench_portfolio")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    reg = ordered(regression())
    reg.to_csv(ANALYSIS / "finbench_comparison_regression.csv", index=False)
    (TABLES / "finbench_regression.tex").write_text(regression_tex(reg), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(reg[["universe", "model", "pred_len", "seeds", "test_years", "MSE", "MSE_sd", "FB_MSE", "delta_MSE_pct",
                   "MAE", "MAE_sd", "FB_MAE", "delta_MAE_pct"]].to_string(index=False))

    port = portfolio()
    if port.empty:
        print(f"\nNo top-{PORTFOLIO_K} portfolios in {PORTFOLIO_ROOT}: run "
              f"`python code/scripts/run_portfolio.py --k {PORTFOLIO_K}` first.")
        return
    port = ordered(port)
    port.to_csv(ANALYSIS / "finbench_comparison_portfolio.csv", index=False)
    (TABLES / "finbench_portfolio.tex").write_text(portfolio_tex(port), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.precision", 3):
        print()
        print(port[["universe", "model", "pred_len", "seeds", "test_years", "CAGR", "CAGR_sd", "FB_CAGR",
                    "index_CAGR", "FB_index_CAGR"]].to_string(index=False))


if __name__ == "__main__":
    main()
