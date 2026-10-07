# Thesis topics: what can go into the final thesis

Every topic the work so far supports or the plan will produce, grouped by the chapter it belongs to. For each topic: **status**, **where the material is**, and the **artefact** (table, figure, paragraph) it can become. Use it to plan chapters and to check that nothing found along the way is lost.

**Status:** ✅ material ready · ◐ partly done · ☐ planned (step of [`THESIS_GUIDE.md`](../THESIS_GUIDE.md)) · ⭘ optional

Last updated: 2026-10-06 (Step 2, equivalence test).

---

## Suggested chapter structure

| # | Chapter | Main sources |
|---|---|---|
| 1 | Introduction | outline, Steps 3–4 numbers |
| 2 | Background and related work | Step 0.2, already drafted (`02_second_chapter.tex`) |
| 3 | Data | Step 1, notebook 01 |
| 4 | Models and a reproducible framework | Step 2, `model_notes.md`, `xaifin` |
| 5 | Explainability methodology | Steps 5–6 |
| 6 | Results | Steps 3, 4, 5, 7, 8 |
| 7 | Portfolio stability and counterfactuals | Steps 4, 9 |
| 8 | Sensitivity to data perturbations | Step 10 |
| 9 | The application | Step 11 |
| 10 | Conclusions, limitations, future work | Step 7 diagnosis |

The template chapters `03_third_chapter.tex` and the template text in `01_introduction.tex` still need replacing.

---

## 1. Introduction

| Topic | Status | Material | Artefact |
|---|---|---|---|
| Motivation: deep models for stock ranking reach low IC; small prediction changes reorder a top-k portfolio. Are explanations of such models trustworthy and comparable? | ◐ | outline; weak-signal evidence in notebook 01 obs. 8 (48–59% positive labels) | opening paragraphs |
| Research questions: (1) which features drive each model; (2) do models agree; (3) is instability **data-driven or model-driven**; (4) do explanations change across periods; (5) how fragile is the allocation | ☐ | outline Steps 1–4 | numbered list of RQs |
| Contributions (see the box at the end of this file) | ◐ | this file | bullet list |
| Thesis outline | ☐ | – | paragraph |

## 2. Background and related work

Already drafted in `latex-thesis/chapters/02_second_chapter.tex` (46 verified references). Topics covered: the stock-forecasting pipeline and task formulation; the six models; post-hoc attribution (gradients, IG, DeepLIFT, SHAP, LIME); time-series XAI (TimeSHAP, TSR, Dynamask, FIT, WinIT); evaluation of explanations (faithfulness, stability, sanity checks); attention as explanation; the Rashomon effect and explanation disagreement; counterfactuals (Wachter, DiCE); XAI in finance and the gap addressed.

| Topic still to add | Status | Material |
|---|---|---|
| Qlib's Alpha158/Alpha360 (paper missing from `docs/papers/`, arXiv 2009.11189) | ☐ | `finbench/Evaluation/features/` |
| FinBench as the benchmark the models and protocol come from | ◐ | FinBench paper §4.1 |
| Resolve the `% TODO` reference to the methodology chapter | ☐ | `02_second_chapter.tex:88` |

## 3. Data

| Topic | Status | Material | Artefact |
|---|---|---|---|
| Five universes (DJI, NASDAQ-100, S&P 500, EURO STOXX 50, STOXX Europe 600); core = DJI, NASDAQ-100, EURO STOXX 50 | ✅ | notebook 01 §1, obs. 2 | **table of universes** (members, tickers with prices, period, volatility) |
| Rolling protocol: 4y train / 1y valid / 1y test, test years 2020–2024; T=20, L=5; seeds 0, 5, 42 | ✅ | guide §2.4–2.5, `config.py` | **table of rolling windows** |
| Survivorship: FinBench trains and tests on the members at the start of the test year | ✅ | `loading.active_tickers` | paragraph (limitation) |
| EU membership history starts 2020-09-30 → no 2020 test year for EU universes | ✅ | notebook 01 obs. 3 | paragraph + footnote in the grid table |
| Label: forward L-day return on adjusted close, daily z-score; weak signal (48–59% positive, ±4–12% range) | ✅ | notebook 01 §6, obs. 8 | **label-distribution figure** per year |
| Feature sets: Alpha158 (157, no VWAP0) and Alpha360 (6 series × 60 lags); market gate features (63 US / 42 EU) | ✅ | notebook 01 §2, `features.catalog` | **feature-catalog table** (family, source, horizon) |
| Feature families, sources and horizons as the common language between the two groups (overlap only on momentum, volume, candle) | ✅ | notebook 01 obs. 1 | table + paragraph; basis for cross-group comparison |
| Alpha158 redundancy: 103–105 correlation groups, 27–29 near-duplicate groups, 20 exactly redundant features (SUMP/N/D, VSUMP/N/D) | ✅ | notebook 01 §7, obs. 9 | **feature-correlation figure**; consequence for SHAP and counterfactuals |
| Missing values negligible (<0.8%; CORD up to 3–4.8%) and filled with 0 = training median | ✅ | notebook 01 §4, obs. 7 | one paragraph |
| **Data quality**: wrong-company tickers (DJI `PRG.US` = PROG Holdings, not P&G, in every DJI test year; `TEN.US`, `TOM.F`), stale prices (`SAABY.US`), near-zero prices (`ORRON.ST`, `DISH.US`), unadjusted corporate actions (`VIV.PA`) | ✅ | notebook 01 §5, obs. 4–5 | **table of data errors** (what the filter removes and why) |
| Data-quality filter: what it discards (0.07–0.9% of samples, 2.3–3.5% for DJI), how it is applied to samples rather than rows, and its effect on label volatility (e.g. STOXX Europe 600 2024: 30% → 4.4%) | ✅ | `data/quality.py`, notebook 01 obs. 6, guide §2.4 | method paragraph + before/after table |
| Volatility regimes: 2020 stressed, 2022 second (peak for NASDAQ-100), 2017 calmest | ✅ | notebook 01 §8, obs. 10 | **volatility timeline**; input for the period chapter |
| STOXX Europe 600 calendar and listing mix (partial cross-sections) | ✅ | notebook 01 obs. 11 | paragraph (why it is not a core universe) |

## 4. Models and a reproducible framework

| Topic | Status | Material | Artefact |
|---|---|---|---|
| Two comparable groups: Alpha158 (MASTER, MATCC, FactorVAE), Alpha360 (HIST, DiscoverPLF, FinFormer); optional LightGBM reference | ✅ | `docs/model_groups.md` | **table of model groups** |
| One page per model: target, inputs, normalization, loss, cross-sectional mixing, built-in explanation signals | ✅ | `docs/model_notes.md` | **summary table of the six models** (already in `model_notes.md`) |
| Implementation choices that change the results, stated briefly in the methodology (not a thesis topic in themselves; background in `model_notes.md`) | ✅ | | one paragraph or a short table |
| — FactorVAE is scored with the mean of its prior path (the prediction it would make without seeing the labels), and its target is a 20-day return | ✅ | `models/factorvae.py` | sentence + caveat wherever FactorVAE is compared |
| — Each model keeps its weights with its original training and selection rule | ✅ | `training/trainer.py` | **table: loss, optimizer, epochs, selection rule per model** |
| — The Alpha360 models read their 360 columns in an order that mixes series and lags, so attributions are reported on the original columns, not on the models' internal time steps | ✅ | `models/finformer.py` (`FINBENCH_ORDER`) | sentence; relevant for the temporal XAI analysis |
| — All runs are seeded (seeds 0, 5, 42), so seed-to-seed differences are controlled | ✅ | `models/base.py` | sentence; matters for the seed-vs-model analysis |
| Decision to train the models as in their reference code (FinBench), and why: results comparable with published numbers | ✅ | guide Step 3 | paragraph (methodological choice) |
| The `xaifin` framework: `ModelAdapter` interface, `DayBatch`, `RunConfig`, results-store layout | ✅ | `models/base.py`, guide §2.2–2.3 | **architecture diagram** (data → adapter → XAI / portfolio / app) |
| Data pipelines reproduced exactly (three ways FinBench builds inputs: own-row windows, Qlib calendar windows with fill, single Alpha360 rows) and verified sample by sample | ✅ | `data/datasets.py`; check scripts | paragraph + "verification" table |
| One input space per group: Alpha360 adapters take the 360 original columns and apply each model's reshape inside → attributions comparable across HIST, DiscoverPLF, FinFormer | ✅ | `models/finformer.py` (`FINBENCH_ORDER`) | paragraph |
| Explaining cross-sectional models: explain a whole day, attribute stock *i* to its own inputs, hold extras (concepts, market cap, graph) fixed | ✅ design | guide §2.3 | method paragraph |
| Gradients in eval mode need cuDNN off for GRUs (engineering note) | ✅ | `ModelAdapter.forward` | footnote |
| Faster, memory-light dataset construction (array indexing vs per-day scans) | ✅ | `datasets.py` | footnote / appendix |
| Equivalence test: adapters vs FinBench's own code, same seed, day order and random state; all six pass | ✅ | notebook 02 | **equivalence table** (initial weights, weights after one epoch, test predictions, metrics) |
| FinFormer's MSE is not comparable (−CCC loss ignores scale): compare it on IC/RankIC | ✅ | notebook 02 obs. 5 | footnote |

## 5. Explainability methodology

| Topic | Status | Material | Artefact |
|---|---|---|---|
| Methods: Saliency, Integrated Gradients, GradientSHAP, DeepLIFT (Captum); KernelSHAP/PermutationSHAP and LIME on feature groups; cross-sectional permutation importance; lag-window occlusion / TimeSHAP-style; intrinsic signals | ☐ | Step 5.1 | **table of methods** (family, resolution, cost) |
| Baseline choice: 0 = training median after the robust z-score; alternative = daily cross-sectional mean | ☐ | guide §2.3, Step 6 | paragraph + sensitivity result |
| Two targets: the raw score vs the **top-k margin** (explains portfolio membership) | ☐ | Step 5, 7 | paragraph |
| Aggregation: per-sample normalization, global importance per feature, lag, family, source, horizon; signed vs absolute | ☐ | Step 5.2 | method paragraph |
| Explaining only rebalance dates (storage) | ☐ | Step 5.1 | footnote |
| Intrinsic signals per model: MASTER's market gate α(m) (a built-in, market-conditioned feature importance), attention maps, FactorVAE's additive α + β·z, HIST's predefined / hidden / individual split, FinFormer's fusion gate | ☐ | `model_notes.md` | comparison post-hoc vs built-in |
| Evaluating explanations: faithfulness (deletion/insertion, AOPC), stability (max-sensitivity, neighbouring days), sanity check (randomized weights), method agreement, baseline sensitivity | ☐ | Step 6 | **evaluation table**; justified choice of the main method |
| Redundant features and attributions: report per correlation group as well as per feature | ◐ | notebook 01 obs. 9 | paragraph |

## 6. Results

| Topic | Status | Material | Artefact |
|---|---|---|---|
| Predictive performance: IC, RankIC, ICIR, MSE per model × universe × year (mean ± std over seeds), compared with the FinBench tables | ✅ seed 42 (notebook 03) | Step 3; first MASTER runs (dji, nasdaq100 y2020) | **performance table**, **IC-per-year plot** |
| How weak the signal is (starting point of the thesis): mean RankIC −0.006 to 0.013 per model, sign changing by year and universe, validation IC barely predicting test IC | ✅ notebook 03 obs. 2-6 | Step 3, notebook 01 obs. 8 | paragraph |
| Effect of the data-quality filter on metrics (`CLEAN_DATA` on/off) | ☐ | Step 3 | small table |
| Top features per model; family-importance profile; lag profile; example local explanation | ☐ | Step 5 | **top-feature tables**, **family figure**, **heatmap T×F** |
| Post-hoc vs intrinsic importance (e.g. MASTER gate vs IG) | ☐ | Step 5 | figure |
| **Cross-model agreement**: Spearman, Kendall, top-k Jaccard, RBO; within groups per feature, across groups per family/horizon | ☐ | Step 7 | **agreement heatmaps** |
| **Variance decomposition** of explanations: seed vs model vs year vs universe | ☐ | Step 7 | **ANOVA-style table** (core result) |
| Consistently dominant features across runs | ☐ | Step 7 | table |
| **Data-vs-model diagnosis** (agree + low IC → data-limited; disagree + similar IC → Rashomon, model-driven; seed ≈ model → explanations unstable) | ☐ | Step 7 | the central argument of the thesis |
| LightGBM + TreeSHAP as "what the data allows" reference | ⭘ | Step 7 | comparison paragraph |
| Score drivers vs portfolio drivers (score vs top-k-margin attributions) | ☐ | Step 7 | figure |
| Period and regime analysis: performance and importance drift vs volatility (COVID 2020, 2022 bear market); monthly attribution series | ☐ | Step 8 | **timeline figure**, regime × model table |

## 7. Portfolio stability and counterfactuals

| Topic | Status | Material | Artefact |
|---|---|---|---|
| Long-only top-k baseline (k = 5 for DJI, 10 otherwise; rebalance every 5 days): CAGR, Sharpe, Sortino, MaxDD, volatility, turnover vs equal-weight and index | ✅ notebook 04 | Step 4 | **portfolio table**, equity curves |
| Seed instability of holdings (Jaccard overlap, turnover between seed portfolios) | ☐ | Step 4 | **seed-overlap figure** |
| Models pick different stocks: two models' portfolios share 10-27% of their stocks (chance: 5-11%); the closest pairs share their feature set (MASTER-FactorVAE, HIST-DiscoverPLF/FinFormer), a first hint that the input data shapes the choices | ✅ notebook 04 obs. 7 | **model-overlap heatmaps** |
| Margins to the k-boundary and **decision fragility** (12-18% of holdings within 0.1 score standard deviations of the cut; seed-noise comparison after stage 4) | ◐ notebook 04 | Steps 4, 9.1 | distribution figure |
| Counterfactuals in feature space (Wachter-style, constraints: z-range, immutable market features, correlated groups) | ⭘ | Step 9.2 A | table of minimal perturbations |
| Counterfactuals in OHLCV space (perturb prices, recompute features): minimal price move that flips membership | ⭘ | Step 9.2 B | distribution per model |

## 8. Sensitivity to data perturbations (*Seconda parte*)

| Topic | Status | Material | Artefact |
|---|---|---|---|
| OHLCV noise injection → Δ IC, Δ portfolio, Δ attributions | ☐ | Step 10 | figure |
| Family ablation with retraining (stronger test of importance than masking) | ☐ | Step 10 | table |
| Delayed data, missing days, universe swap | ⭘ | Step 10 | table |
| Retraining the Alpha360 models with chronologically ordered inputs: does it change accuracy or explanations? | ⭘ | `model_notes.md` | experiment |
| Effect of the data errors themselves (`CLEAN_DATA` off vs on) on explanations | ⭘ | Step 10 | figure |

## 9. The application

| Topic | Status | Material | Artefact |
|---|---|---|---|
| Streamlit app: purpose and design (pages grow with the steps, charts shared with the notebooks via `xaifin/viz.py`, caching) | ◐ | `app/Home.py`, `app/pages/1_Data.py` | screenshots |
| Pages: Data ✅, Models, Explain, Compare, Over time, Portfolio | ◐ | Step 11 table | one screenshot per page |
| Load any checkpoint from its `config.json` and explain it on demand (GPU) | ☐ | Step 11 | walkthrough |

## 10. Conclusions, limitations, future work

| Topic | Status | Material |
|---|---|---|
| Answer to each research question; the data-vs-model verdict | ☐ | Step 7 |
| **Limitations:** survivorship bias of the constituent rule; implementation choices kept from the reference code (FactorVAE's 20-day target, the Alpha360 input order, per-model selection rules); remaining data errors after the filter (`ORRON.ST`, Signature Bank); attributions on correlated features; no EU 2020 test year; single horizon (T=20, L=5) | ◐ | `model_notes.md`, notebook 01 |
| **Future work:** retrain with a common target horizon and chronological Alpha360 inputs; other horizons (5/1, 60/20); backfilled EU membership; cross-stock attribution (how stock *j* influences *i*); more universes | ☐ | – |
| Reproducibility statement: code, seeds, git hash in every `config.json`, results-store layout | ◐ | guide §2.2, §2.4 |

---

## Contributions (draft)

1. **A unified, verified framework** (`xaifin`) that wraps six heterogeneous stock-ranking models behind one interface and reproduces FinBench's data pipelines exactly, so that explanations can be computed and compared across architectures.
2. **A systematic comparison of explanations** across models, seeds, periods and universes, with evaluated XAI methods, and a diagnosis of whether instability comes from the data or the architecture. *(Steps 5–7)*
3. **A link from explanations to allocation**: attributions of portfolio membership, margins and counterfactual fragility of top-k portfolios. *(Steps 4, 7, 9)*
4. **An interactive application** to load models, inspect attributions and compare architectures. *(Step 11)*

## Figures and tables checklist

Ready now: universes table · rolling-window table · model-groups table · six-models summary table · feature catalog · feature-correlation figure · label distribution · volatility timeline · data-error table · data-filter before/after · training-setup table (loss, optimizer, selection rule per model).

Ready since Step 2: equivalence table.

Planned: performance table · IC per year · XAI methods table · XAI evaluation table · top-feature tables · family-importance figure · local heatmap · agreement heatmaps · variance decomposition · dominant features · regime timeline · portfolio table · seed-overlap figure · margin/fragility distribution · counterfactual distributions · app screenshots · framework diagram.
