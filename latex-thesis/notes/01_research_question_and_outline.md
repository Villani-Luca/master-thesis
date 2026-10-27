# 1. Research question, structure and scope

Sources: `docs/papers/Outline Tesi.pdf`, `THESIS_GUIDE.md` (top and §Step 12), `docs/stato_avanzamento.md`, `PIANO.md`.

## Research question

Stock-prediction models have weak predictive power (IC around 0.0x, direction accuracy near 50%),
and that weakness carries over into long-only top-k portfolios. The thesis asks:

1. **How do the input features drive the predictions** of state-of-the-art deep stock-forecasting
   models, and **how does that turn into allocation decisions** (which stocks enter the top-k)?
2. **Does the instability come from the data or from the model design?** If models with different
   architectures agree on which features matter while all having low IC, the limit is in the data;
   if they disagree with similar IC, the architecture decides which signal is used (Rashomon effect);
   if seeds of the same model disagree as much as different models, explanations themselves are not
   stable enough to conclude anything about architectures.

Expected outcome of the outline: a framework that applies XAI methods to the models and saves
results and metrics, plus a graphical application to load models, show attributions and compare
architectures.

## Motivation already backed by results (for the Introduction)

- The six models reach a test RankIC between −0.009 and +0.012 on average (file 04).
- Two seeds of the same model share only 17–46% of their top-k portfolio, barely more than two
  different models (10–27%) (file 05).
- The yearly return of one model can change by up to 15 points with the seed (file 05).
- So: small changes in training → different predictions → different portfolios. Explaining what the
  models rely on is the way to understand that instability.

## Proposed structure (to agree with the supervisor)

Current files in `latex-thesis/chapters/`: `01_introduction` (template text, ~60 words),
`02_second_chapter` (Background and Related Work, drafted, ~3700 words, 46 verified references),
`03_third_chapter` (template placeholder "Tables and Images"), `05_conclusions` (empty).

| # | Chapter | Content | Notes file |
|---|---|---|---|
| 1 | Introduction | weak signal → unstable allocation; research question; contributions; structure | 01 |
| 2 | Background and Related Work | done in draft: forecasting pipeline, FinBench, six models, XAI (perturbation, surrogate, gradient, attention), time-series XAI, evaluation of explanations, counterfactuals, XAI in finance, research gap | – |
| 3 | Data and Models | universes, features and families, data quality; the two model groups; FinBench issues and the fixes | 02, 03 |
| 4 | Methodology | rolling protocol and selection rules; portfolio construction; the framework; XAI methods, targets, aggregation, evaluation; agreement and variance decomposition | 04, 05, 06, 07 |
| 5 | Results | predictive performance and FinBench comparison; portfolios and their fragility; attributions; evaluation of the explanations; cross-model comparison and the data-vs-model diagnosis; volatility periods | 04, 05 (+ XAI results to come) |
| 6 | The Application | pages, workflow, screenshots | 06 |
| 7 | Conclusions | diagnosis, limitations, future work | 01 |

Possible contributions to state in the Introduction (all already done unless noted):
- a reproduction of six FinBench regression models behind one interface, **verified equivalent to
  FinBench's code**, with ten documented implementation issues of the benchmark (two fixed:
  FactorVAE's test-time label leakage and random readout);
- a data-quality filter for the FinBench datasets (wrong-company tickers, stale and bad prices);
- a measure of how weak and seed-dependent the signal and the portfolios are;
- (to come) an XAI framework that explains both the score and top-k membership, and a cross-model,
  cross-seed comparison of explanations;
- (to come) the application.

## Scope (deadline 18 November 2026) and future work

In the thesis: Steps 5 (IG/Saliency, permutation importance by family, attention where available;
SHAP/LIME on DJI 2020 as a check), 6 (reduced), 7 (in full), 8 (light, yearly), 9.1 (prediction-space
fragility), application pages Explain and Compare.

**Future work** (for the Conclusions), cut for time:
- SHAP/LIME on the whole grid;
- input-space counterfactuals (gradient-based in feature space; black-box on OHLCV with recomputed
  Alpha features);
- data-perturbation tests: noise on prices, family ablation with retraining, delayed data, universe swap;
- monthly attribution series inside each year; a LightGBM/TreeSHAP reference;
- retraining the Alpha360 models with the chronological column layout (FinBench issue 4, file 03);
- the explanation-stability and baseline-sensitivity checks of Step 6.
