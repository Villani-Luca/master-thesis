# 4. Rolling training and predictive performance

Sources: notebook `code/notebooks/03_rolling_training.ipynb` (*Observations*),
`code/results/analysis/metrics_summary.csv` (one row per run, all 252 runs),
`code/results/analysis/finbench_comparison_regression.csv` and `latex-thesis/tables/finbench_regression.tex`
(by `code/scripts/compare_finbench.py`), `xaifin/config.py`, `xaifin/training/`, `THESIS_GUIDE.md` Step 3.

## Protocol (FinBench, paper §4.1)

- Rolling windows: **4 years of training, 1 of validation, 1 of test**; test years 2020–2024
  (window for test year Y: training Y−5 … Y−2, validation Y−1, test Y). SX5E/SXXP: 2021–2024.
- (T, L) ∈ {(5, 1), (20, 5), (60, 20)}; the first grid uses **T = 20, L = 5**. Alpha360 models use an
  effective lookback of 1 because the 360-vector already covers 60 days.
- Seeds 0, 5, 42. Every generator is seeded (FinBench seeds only some of them, issue 8).
- Data-quality filter on (file 02).

## Selection rules: each model as in FinBench (decision 2026-10-06, "Option A")

So that results stay comparable with FinBench's tables, every model keeps its own training and its
own rule for the final weights (`xaifin.training.trainer.SELECTION_RULES`):

| Model | Rule | Median epochs run / kept |
|---|---|---|
| MASTER | stop when training loss ≤ 0.95, keep the last weights | 11 / 10 |
| MATCC | train 70 epochs, keep the last | 70 / 69 |
| FactorVAE | keep the lowest validation loss (30 epochs) | 30 / 1 |
| HIST | best validation IC of the weights averaged over the last 5 epochs, early stopping | 44 / 13 |
| DiscoverPLF | as HIST, from epoch 20 | 50 / 29 |
| FinFormer | best validation IC pooled over all stock-days, early stopping | 14 / 3 |

Every run logs the validation metrics of every epoch (`train_history.csv`), so other rules can be
studied without retraining.

## The grid

Stages (each checked before the next): MASTER × DJI × 2020; 6 models × DJI × 2020; 6 models × 3
universes × 2020–2024 × seed 42 (84 runs); seeds 0 and 5 (168 runs). **252 runs, no failures, ~88
GPU hours on one RTX 5090**; FactorVAE alone 47 hours (67 minutes per run, FinBench's 96 attention
layers run sequentially for every day); per run MASTER 2 min, FinFormer 5, HIST 11, MATCC 16, DiscoverPLF 24.

Extended grid (running on an external H200, back ~4–9 Nov): all 5 universes × L ∈ {1, 5, 20} × 3
seeds = **1242 runs** (~39 GPU-days serial; ~13 days with 8 runs sharing one GPU).

## Metrics

On the test year, with the labels the model is scored on (daily z-score of the forward return):
IC and RankIC = daily cross-sectional Pearson / Spearman correlation between prediction and label,
averaged over test days; ICIR / RankICIR = mean / std of the daily values; MSE, MAE.
Significance: labels of consecutive days overlap (5-day returns), so the t-statistic is
RankICIR × √(test days / L), counting only non-overlapping periods.

## Results (252 runs, T = 20, L = 5)

Mean test RankIC over years and seeds:

| Model | DJI | NASDAQ-100 | SX5E | All |
|---|---|---|---|---|
| MASTER | 0.0150 | 0.0100 | −0.0152 | 0.0046 |
| MATCC | 0.0157 | 0.0143 | −0.0073 | 0.0086 |
| FactorVAE | 0.0149 | −0.0331 | −0.0092 | −0.0092 |
| HIST | 0.0300 | 0.0063 | −0.0051 | 0.0115 |
| DiscoverPLF | 0.0175 | 0.0046 | −0.0083 | 0.0055 |
| FinFormer | 0.0237 | 0.0024 | 0.0002 | 0.0093 |

(IC has the same pattern: −0.008 to +0.010 overall.)

Findings (notebook 03, *Observations*):
1. **The signal is weak everywhere.** Each model's average RankIC is between −0.009 (FactorVAE) and
   +0.012 (HIST). RankIC > 0 in 40% (FactorVAE) to 67% (MATCC, HIST) of runs, vs ~50% with no skill.
   Counting non-overlapping weeks, |t| ≥ 2 in at most 1 of 42 runs per model, except FactorVAE (14 of
   42, in both directions).
2. **The universe matters more than the model.** On DJI every model is positive (0.015–0.030); on
   NASDAQ-100 between −0.033 and +0.014; on SX5E all at or below 0 (−0.015 to +0.000).
3. **No model wins consistently.** The best model changes from cell to cell; five of six top at least
   one (model, year) cell; the best run of a cell reaches RankIC 0.005–0.096.
4. **Training luck is about half the model differences.** Within a universe and year, the std of
   RankIC across one model's three seeds is 0.010–0.014, against 0.020–0.023 across the six models;
   in 33–46% of the cells the three seeds do not agree on the sign of RankIC. FactorVAE's seeds agree
   most (14%) because its rule keeps an almost untrained model.
5. **The selection rules choose almost blindly.** Correlation between the validation IC of the kept
   epoch and the test IC: −0.19 to 0.15 by model; validation scores are optimistic (0.006–0.037 vs
   −0.008 to 0.010 on test). FactorVAE keeps epoch 0 or 1 of 30 in almost every run.

Conclusion to carry into the XAI part: the six architectures extract a small, unstable signal whose
sign changes with period, universe and seed.

## Comparison with FinBench (paper Table 2: MSE and MAE only)

FinBench publishes no IC for these models. Ours averaged over test years, then seeds (L = 5):

| | DJI MSE (ours / FB) | NASDAQ-100 | SX5E |
|---|---|---|---|
| MASTER | 0.980 / 1.249 | 1.015 / 1.017 | 1.004 / 1.097 |
| MATCC | 1.491 / 1.130 | 1.731 / 1.713 | 1.644 / 1.495 |
| FactorVAE | 0.888 / 0.971 | 0.903 / 0.918 | 0.944 / 0.960 |
| HIST | 0.985 / 0.997 | 1.035 / 1.030 | 1.068 / 1.113 |
| DiscoverPLF | 1.187 / 1.211 | 1.154 / 1.104 | 1.212 / 1.149 |
| FinFormer | 2.467 / 1.657 | 2.403 / 1.458 | 2.604 / 1.430 |

- Same within ~5% in 10 of 18 pairs (HIST and DiscoverPLF everywhere, most of NASDAQ-100); better in 3
  (MASTER on DJI −21% and SX5E −8%; FactorVAE on DJI −8.5%, even without FinBench's leak); worse in 5
  (FinFormer +49% to +82%, its −CCC loss ignores scale; MATCC on DJI +32%, SX5E +10%).
- FinBench's numbers mix unseeded runs (issue 8) and, for FactorVAE, the leaky readout (issue 1), and
  SX5E's FinBench years include 2020; only the range is comparable.
- **Pending**: the portfolio comparison with FinBench's Table 7 (top-5 CAGR) needs the top-5
  portfolios (`run_portfolio.py --k 5`, then `compare_finbench.py`); the index returns already match
  (DJI 0.087 vs FinBench 0.08).

## To update when the H200 grid is back

All tables for 5 universes × 3 horizons; FinBench comparison for L = 1 and 20 and for S&P 500/SXXP
(Table 2 has them); does the weak-signal picture hold on the large universes and other horizons?
