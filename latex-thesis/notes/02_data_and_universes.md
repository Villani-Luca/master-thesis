# 2. Data and universes

Sources: notebook `code/notebooks/01_universes_and_data.ipynb` (sections and *Observations*),
`xaifin/config.py`, `xaifin/data/features.py`, `xaifin/data/quality.py`, `THESIS_GUIDE.md` Step 1.
App page `1_Data` shows the same tables and charts.

## Universes

Five investment universes from FinBench, study period 2015–2024:

| Universe | Region | Members at start of each test year | Price tickers (incl. past members) | Top-k used |
|---|---|---|---|---|
| DJI | US | 30 | 43–44 | 5 |
| NASDAQ-100 | US | 101–104 | | 10 |
| S&P 500 | US | 503–505 | | 10 |
| EURO STOXX 50 (SX5E) | EU | 50–51 | | 10 |
| STOXX Europe 600 (SXXP) | EU | 600–615 | 742–749 | 10 |

- FinBench trains on the **constituents active at the start of each test year**.
- **The EU membership history starts on 2020-09-30**, so at 2020-01-01 no stock passes the rule:
  the 2020 test year is empty for SX5E and SXXP, whose rolling grid is 2021–2024.
- Core universes of the first grid: DJI, NASDAQ-100, SX5E. S&P 500 and SXXP join with the H200 grid.
- SXXP mixes calendars and listings: exchanges close on different holidays, and some members are
  priced through US OTC (`.US`) or Frankfurt (`.F`) listings, so on some days the cross-section is partial.

## Features: two groups, little overlap

| Group | Features | Models | Input per stock |
|---|---|---|---|
| A: Alpha158 | 157 engineered factors (Qlib's 158 minus `VWAP0`; FinBench has no VWAP input) + market gate features for MASTER/MATCC (63 US, 42 EU) | MASTER, MATCC, FactorVAE | `[T = 20, 157 (+63/+42)]` |
| B: Alpha360 | 6 raw normalized series (CLOSE, OPEN, HIGH, LOW, VOLUME, VWAP) × 60 lags | HIST, DiscoverPLF, FinFormer | 360-vector |

- Feature catalog (`xaifin.data.features.catalog`): every feature has a **family** (8: momentum,
  trend, position, volatility, candle, volume, price_volume, market), a **source** (close, ohlc,
  volume, price_volume, market) and a **horizon** (short ≤ 5 days, medium 10–20, long 30–60). Names
  checked against the CSV headers of all five universes.
- 296 of the 360 Alpha360 columns are momentum-type (past prices over today's close). The two sets
  share only momentum, volume and candle features, so **comparisons across the groups use family,
  source and horizon**, never single features.
- Features are normalized with a robust z-score (median/MAD) fit on the training window, clipped to
  [−3, 3]; missing values are filled with 0 (the training median).

## Label

FinBench's forward return on the adjusted close over the horizon L:
r_{i,t} = (p_{i,t+L} − p_{i,t}) / p_{i,t}, then z-scored cross-sectionally each day by the models.
Exception: FactorVAE's label uses T instead of L (a 20-day return with T = 20), kept as FinBench (file 03).

After the data-quality filter the 5-day return has a median between −0.2% and 0.7%, a 5th–95th
percentile range of ±10–12% in 2020 and ±4–8% in most other years; **the share of positive labels
is 48–59%**, so guessing "up" is barely better than a coin flip.

## Data quality (a contribution: FinBench keeps these problems)

Problems found:
- **Tickers mapped to the wrong company**, in test universes: DJI `PRG.US` is listed as Procter &
  Gamble but its prices are PROG Holdings ($12–64 vs P&G's $51–175), in **every** DJI test year;
  SXXP `TEN.US` (Tenaris) is Tsakos Energy Navigation (2021–2023); `TOM.F` (Tomra) is Toyota Motor
  (2021–2022).
- **Bad prices that reach a model**: SXXP `SAABY.US` constant at 992,237 for five years then ~11
  (a −99.999% label in 2024); `ORRON.ST` down to 0.0002 with 50 labels ≥ +100%; S&P 500 `DISH.US`
  recorded at 0 after its merger (−100% label in 2023); post-failure prices of Signature Bank
  (2023); reused tickers repeating a price for years; SX5E `VIV.PA` at 0.044 instead of 4.11 on 2015-12-31.

**The filter** (`xaifin/data/quality.py`, decided 2026-09-29): a sample (stock i, day t) is discarded
when a problem falls in the days it reads, from the first price behind its features to the end of its
label: wrong-company ticker; stale price (same adjusted close for ≥ 20 consecutive days); price below
0.01; one-day jump beyond ×3 or ÷3. It removes samples, not file rows, so sequences never join days
across a gap; it applies to training, validation and test (`config.CLEAN_DATA`; off only to
reproduce FinBench in the equivalence test).

Effect: removes 0.07% (SX5E) to 0.9% (SXXP) of the samples, 2.3% for DJI (all `PRG.US`). It brings the
yearly standard deviation of the label back in line with market volatility (SX5E 2015: 73% → 3.9%;
SXXP 2024: 30% → 4.4%; NASDAQ-100 2022: 17.5% → 7.0%). Residual known errors: `ORRON.ST` penny prices
above 0.01 (30 labels, 2021–2022) and 11 post-failure labels of Signature Bank (2023).

Missing values are negligible: under 0.8% per feature, except NASDAQ-100 (up to 3%) and SXXP (up to
4.8%), both in CORD (price-volume change correlation).

## Redundancy among Alpha158 features

Absolute Spearman correlation pooled over every Wednesday of the study period, complete linkage at
|ρ| ≥ 0.9: the 157 features form 103–105 groups, and 80–82 features sit in 27–29 groups of
near-duplicates, consistently across universes. SUMP/SUMN/SUMD and VSUMP/VSUMN/VSUMD are exact
rescalings (SUMN = 1 − SUMP, SUMD = 2·SUMP − 1): **20 features add no information**.
Consequence for XAI: SHAP and permutation importance split or blur credit within a group, so
attributions must be reported per group/family as well as per feature (decision in file 07);
counterfactuals would have to move a group together.

## Market volatility (input for the period analysis)

Annualized 20-day realized volatility of the index (`std_price_change_20_*` × √252). **2020** is the
stressed year (DJI 28%, S&P 500 27%, SX5E 27%, SXXP 23%, about twice a calm year), then **2022**,
the peak for NASDAQ-100 (32%). 2017 is the calmest (DJI 6.5%). Test years 2020 and 2022 are the
high-volatility regimes.

## For the thesis

- Tables available in notebook 01: model groups; universes (members, tickers, stock-days, share
  discarded, label IQR, index volatility); wrong tickers; filter effect per year.
- Figures: coverage over time, missing values by family, label distribution per year, Alpha158
  correlation dendrogram/heatmap, index volatility per year.
- To state: why the filter is a deliberate deviation from FinBench, and that it only removes samples.
