# 7. XAI: design decisions taken so far (not implemented yet)

Sources: `THESIS_GUIDE.md` Steps 5–9, `PIANO.md` (decisions D1–D6, 2026-10-22), `docs/model_notes.md`
(§ "What this means for the XAI framework"). Everything here is a **plan**: write the methodology from
it, then check against the code once Steps 5–7 are done.

## What is explained

- The trained checkpoints, unchanged: XAI is inference only (no retraining).
- **Two targets** (decision D4):
  - `pred`: the model's score for stock i on day t;
  - `topk_margin`: score minus the k-boundary score, i.e. what keeps the stock in (or out of) the
    portfolio. This links features to **allocation**, the second half of the research question.
- **Days**: only rebalance dates, ~50 per year, the same L = 5 dates for every horizon.
- **Cross-sectional models**: explain a whole day at once; attribute stock i's output to its own
  inputs, the other stocks and the extra inputs (concept matrix, market cap, adjacency) held fixed.
- **Input space**: Alpha158 `[T, F]` per stock; Alpha360 on the **original 360 CSV columns**, mapped to
  (series, lag) afterwards (FinBench issue 4).

## Methods (decision D3)

| Method | Resolution | Scope |
|---|---|---|
| Integrated Gradients (+ Saliency) | full T × F per stock and day; baseline 0 = training median after the robust z-score | every run |
| Cross-sectional permutation importance | per feature **family** (decision D5): shuffle a family across stocks within each day, measure ΔRankIC and the change of top-k membership | every run |
| Attention / built-in signals | MASTER's gate α(m) and attention, MATCC attention, FactorVAE α + β·z, HIST three-way split, FinFormer fusion gate — where the model exposes them without major surgery | supporting evidence only ("attention is not explanation") |
| KernelSHAP / LIME | feature families as players | DJI, test year 2020, all six models: a check of IG |

## Aggregation

Per sample, normalize |a| / Σ|a| so models with different output scales are comparable; then global
importance per feature (sum over T), per lag (sum over F), per family, with mean, dispersion over
dates and stocks, and the signed mean (direction). Rank the features per run (`global.parquet`).
Across the two groups compare only families, sources and horizons (file 02).

## Evaluation of explanations (reduced Step 6)

- Sanity check (Adebayo et al., 2018): randomize the weights; attributions must change.
- Agreement between methods (IG, permutation, SHAP, LIME) on the same run.
- Faithfulness: deletion curve (mask the top-attributed features, compare with random masking, AOPC).
- Cut: max-sensitivity under input noise, baseline sensitivity (future work).

## Cross-model diagnosis (Step 7, the central result)

- Agreement between importance rankings: Spearman ρ, Kendall τ, top-k Jaccard, rank-biased overlap;
  at feature level within a group, at family/source/horizon level across groups.
- Variance decomposition of the importance vectors over model, seed, year, universe (and horizon if
  the L = 1/20 attributions arrive in time).
- Diagnosis rules (write them explicitly):
  - models agree and all have low IC → **data-driven** instability;
  - models disagree with similar IC → **model-driven** (Rashomon effect);
  - seed disagreement ≈ model disagreement → explanations not stable enough to conclude.
- Repeat with `topk_margin`: do the features that drive the portfolio differ from those that drive
  the score?
- Seeds: available now only for MASTER (and a few MATCC/FactorVAE runs); for all models once the H200
  grid returns the seed-0/5 checkpoints.

## Already known, useful for interpreting the results

- Alpha158 is highly redundant (20 exact duplicates, 27–29 near-duplicate groups): per-feature credit
  is unreliable, families are the unit (file 02).
- Portfolios of models sharing a feature set overlap more (file 05, finding 6): a prior to test.
- FactorVAE's selection rule keeps an almost untrained model (epoch 0–1): its explanations describe an
  early-training model; say so.
- FactorVAE predicts a 20-day return (issue 3) and Alpha360 models read non-chronological sequences
  (issue 4): architecture differences are partly implementation differences.

## Periods and fragility (light Steps 8 and 9.1)

- Per test year, 2020 and 2022 (high volatility) against the calmer years: IC, portfolio returns,
  family shares, year-to-year drift of importance.
- Decision fragility: share of holdings whose margin to the k-boundary is smaller than the seed noise
  of the predictions.
