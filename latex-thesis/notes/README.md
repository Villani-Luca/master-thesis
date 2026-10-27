# Writing notes: what has been done so far

Material for writing the thesis, collected on 2026-10-22 from the code, the notebooks
(`code/notebooks/01`–`04`), `docs/` and the results tables (`code/results/analysis/`). Each file
covers one block of the thesis, states the facts and numbers with their source, and says where they
go in the text. The notes are in English, like the thesis.

| File | Content | Chapter it feeds |
|---|---|---|
| [01_research_question_and_outline.md](01_research_question_and_outline.md) | research question, proposed chapter structure, scope and future work | Introduction, structure of the whole thesis, Conclusions |
| [02_data_and_universes.md](02_data_and_universes.md) | universes, Alpha158/Alpha360, feature catalog, data quality, labels, redundancy, volatility | Methodology: data |
| [03_models_and_framework.md](03_models_and_framework.md) | the six models, the two groups, FinBench implementation issues, the unified framework and the equivalence test | Methodology: models; Background (models) |
| [04_training_and_predictive_performance.md](04_training_and_predictive_performance.md) | rolling protocol, selection rules, grid, predictive results, comparison with FinBench | Methodology: protocol; Results: predictive performance |
| [05_portfolio_baseline.md](05_portfolio_baseline.md) | long-only top-k rule, validation against FinBench, returns, fragility, overlap between models and seeds | Methodology: allocation; Results: portfolios |
| [06_software_and_compute.md](06_software_and_compute.md) | the `xaifin` package, results store, scripts, app, reproducibility, compute | Methodology: implementation; The application |
| [07_xai_design_decisions.md](07_xai_design_decisions.md) | decisions taken for the XAI part (not yet implemented) | Methodology: XAI methods |

## Status at 2026-10-22

- **Done** (Steps 0–4 of `THESIS_GUIDE.md`): environment and literature, model groups and data
  analysis, unified framework with six verified adapters, rolling training of 252 runs (6 models ×
  DJI, NASDAQ-100, EURO STOXX 50 × test years 2020–2024 × 3 seeds, L = 5), long-only top-k portfolios
  validated against FinBench, app pages Data / Models / Portfolio, chapter 2 (Background) drafted.
- **Running**: the extended grid (1242 runs: also S&P 500 and STOXX Europe 600, horizons L = 1 and
  20) on an external H200, back around 4–9 November. The numbers in files 04 and 05 will then be
  extended to all universes and horizons.
- **Not started**: XAI (Steps 5–8), application pages for explanations, chapters on methodology and
  results.
- **Caveat on the seeds 0 and 5**: their metrics are in `code/results/analysis/metrics_summary.csv`
  (all 252 runs), and the notebooks 03–04 used them, but the checkpoints and predictions of those
  runs were lost. The H200 grid retrains them. Training is seeded and deterministic, but a different
  GPU can change floating-point rounding, so the seed-0/5 numbers may move slightly when they are
  recomputed: check before quoting final numbers.

## Conventions for the text

- Universe names: DJI (Dow Jones Industrial Average), NASDAQ-100, S&P 500, EURO STOXX 50 (SX5E),
  STOXX Europe 600 (SXXP). Keep one spelling throughout.
- Horizon `L` (prediction length) and lookback `T` (sequence length): (T, L) ∈ {(5, 1), (20, 5), (60, 20)}.
- Metrics: IC, RankIC (daily cross-sectional Pearson / Spearman correlation, averaged over test
  days), ICIR, RankICIR, MSE, MAE; portfolio: CAGR, Sharpe, Sortino, max drawdown, volatility, turnover.
- Model names as in the papers: MASTER, MATCC, FactorVAE, HIST, DiscoverPLF, FinFormer.
- Figures exported from notebooks go to `latex-thesis/images/results/` with fixed names.
