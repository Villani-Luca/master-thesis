# 5. Portfolio baseline (long-only top-k)

Sources: notebook `code/notebooks/04_portfolio_baseline.ipynb` (*Observations*),
`code/results/analysis/portfolio_summary.csv`, `portfolio_margins.csv`, `xaifin/portfolio/topk.py`,
`backtest.py`, `THESIS_GUIDE.md` Step 4. App page `6_Portfolio`.

## Construction (FinBench's long-only evaluation)

- Every L = 5 business days (FinBench's `freq = f'{pl}B'`), take the model's latest prediction and buy
  the **k highest-scored stocks in equal weights**: k = 5 for DJI, 10 for the other universes. Hold
  until the next rebalance.
- A stock earns its open-to-close return on its first day (bought at the open), then close-to-close.
- Metrics (quantstats formulas, rf = 0): CAGR, Sharpe, Sortino, max drawdown, annualized volatility,
  turnover (average share of the portfolio replaced at each rebalance).
- Benchmarks on the same days: **equal weight** of all candidate stocks (no skill at choosing among
  them) and the **index**.
- **Margin to the k-boundary** of every candidate: score_i − score_(k+1) for held stocks,
  score_k − score_i for excluded ones; `margin_z` = margin in standard deviations of that day's scores,
  comparable across models. Reused for decision fragility (Step 9.1) and as an XAI target (file 07).
- **Overlap** between two portfolios: Jaccard (shared stocks / stocks held by either) averaged over
  common weeks. Random reference for k stocks out of N: about k / (2N − k), i.e. ~0.09 DJI, 0.05
  NASDAQ-100, 0.11 SX5E.

## Validation against FinBench

On MASTER, DJI, seed 42, 2020–2024, computed both with FinBench's own code
(`Evaluation/portfolio/transforms.py`, `returns.py`, loaded read-only) and with `xaifin`: **same
holdings in all 260 periods, daily returns within 3e-17, identical metrics** against FinBench's
quantstats. Index returns also match FinBench's paper (DJI 0.087 vs 0.08).

## Results (54 portfolios: 6 models × 3 universes × 3 seeds; T = 20, L = 5)

Seed 42, CAGR (Sharpe):

| | DJI (k = 5) | NASDAQ-100 (k = 10) | SX5E (k = 10) |
|---|---|---|---|
| MASTER | 7.8% (0.43) | 9.0% (0.43) | 12.0% (0.68) |
| MATCC | 8.6% (0.47) | 16.2% (0.63) | 11.9% (0.67) |
| FactorVAE | 11.4% (0.57) | −5.7% (0.02) | 5.3% (0.37) |
| HIST | 4.7% (0.31) | 9.4% (0.43) | 9.7% (0.57) |
| DiscoverPLF | 8.0% (0.44) | 5.9% (0.34) | 10.7% (0.62) |
| FinFormer | −0.1% (0.11) | 8.0% (0.39) | 6.8% (0.44) |
| Equal weight | 5.9% (0.39) | 9.6% (0.51) | 10.9% (0.72) |
| Index | 8.7% (0.51) | 20.2% (0.84) | 8.2% (0.54) |

Findings (notebook 04, *Observations*):
1. **Few portfolios beat holding every stock.** DJI: four of six beat equal weight; NASDAQ-100: only
   MATCC, and none reaches the index (driven by a few large tech stocks); SX5E: MASTER and MATCC by
   about one point.
2. **Risk-adjusted, the models rarely add anything.** 5–10 stocks are more volatile (24% vs 19% on
   DJI, 32–36% vs 24% on NASDAQ-100) with deeper drawdowns. Higher Sharpe than equal weight: four of
   six on DJI, only MATCC on NASDAQ-100 (0.63 vs 0.51), none on SX5E.
3. **Returns follow the market.** Every portfolio but one (MASTER on DJI, +0.6%) loses in 2022 (−1.7%
   to −47%), every one gains in 2023.
4. **High turnover**: 53–81% of the portfolio replaced per weekly rebalance for five models; 27–33% for
   FactorVAE (its 20-day target changes its ranking slowly).
5. **The choices are fragile**: the median held stock is 0.24–0.64 standard deviations of the day's
   scores from dropping out; 11–26% of the holdings are within 0.1 and 26–52% within 0.25.
6. **Different models pick different stocks**: two models (same seed) share on average 18% (DJI), 15%
   (NASDAQ-100), 18% (SX5E) of their stocks, pairs 10–27%, against ~9%, 5%, 11% at random. The closest
   pairs share their feature set: MASTER–FactorVAE (Alpha158), HIST–DiscoverPLF and HIST–FinFormer
   (Alpha360): **a first hint that the input data shapes the choices** (to test in Step 7).
7. **The same model with another seed picks different stocks too**: two seeds share 17–46% (MATCC
   17–28%, FinFormer 26–32%, MASTER 25–34%, DiscoverPLF 33–35%, HIST 36–43%, FactorVAE 38–46%), only
   somewhat more than two different models. Retraining with another seed replaces half to three
   quarters of the portfolio.
8. **And earns a different return**: the yearly growth of one model's three seeds differs by up to 15
   points (FinFormer on DJI: −2.1%, −0.1%, +12.8%), 6–11 points for most cells. A ranking of models by
   return can flip with the seed: **any comparison of architectures has to average over seeds and
   report the spread.**

## For the thesis

- Tables: CAGR/Sharpe/turnover per model and universe with benchmarks; seed overlap; model-pair overlap.
- Figures: equity curves, holdings heatmap, margin histogram, overlap heatmap (notebook 04, app).
- Point 6 and 7 together are the bridge to the XAI chapters: they motivate explaining **top-k
  membership**, not only the score.
- To update with the H200 grid: all universes and horizons (rebalance every L days: daily for L = 1,
  every 20 days for L = 20), the FinBench Table 7 comparison (top-5).
