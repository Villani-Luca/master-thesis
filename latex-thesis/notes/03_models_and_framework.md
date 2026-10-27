# 3. Models and the unified framework

Sources: `docs/model_notes.md` (per-model notes and the ten FinBench issues), `docs/model_groups.md`,
notebook `code/notebooks/02_model_adapters.ipynb`, `xaifin/models/`, `THESIS_GUIDE.md` Steps 1–2.
Chapter 2 already describes the six models from the papers; this file is about how they are used here.

## Why these six models

FinBench groups its models by input data (`docs/model_groups.md`). The two **alpha groups** are the
regression models of the outline and share data and protocol, so their explanations can be compared:

| | MASTER | MATCC | FactorVAE | HIST | DiscoverPLF | FinFormer |
|---|---|---|---|---|---|---|
| Venue | AAAI 2024 | CIKM 2024 | AAAI 2022 | arXiv 2021 | TKDE 2024 | IEEE BigData 2023 |
| Features | Alpha158 + market | Alpha158 + market | Alpha158 | Alpha360 | Alpha360 | Alpha360 |
| Extra inputs | – | – | – | stock→concept matrix, market cap | as HIST | sector/industry adjacency |
| Backbone | Transformer | RWKV | GRU + VAE | 2-layer GRU | GRU/dilated + VAE | GRU + sparse static-dynamic transformer |
| Cross-sectional mixing | inter-stock attention | inter-stock attention + market trend | attention in factor predictor | concept graphs | concept graphs | static + dynamic spatial attention |
| Loss (FinBench) | MSE | MSE | MSE recon. + KL | MSE | recon. + pred. + KL | −CCC |
| Built-in explanation | market gate α(m), attention | attention, trend vs fluctuation | ŷ = α + β·z | predefined + hidden + individual | signal-component weights | fusion gate, spatial attention |

**All six are cross-sectional**: a stock's prediction depends on the other stocks of that day, so
explanations are computed for a whole day, attributing each stock's output to its own inputs with
the others fixed. Four models have a built-in decomposition (MASTER's gate, FactorVAE's α + β·z,
HIST's three-way split, FinFormer's fusion gate) that can serve as a check on post-hoc attributions.

## Implementation issues found in FinBench (a contribution)

Found by reading FinBench's code and, for the layout issue, applying each reshape to the real column
names. **Decision (2026-09-29): train exactly as FinBench; fix only the prediction readout and file
paths**, i.e. nothing that changes what the model learns.

| # | Issue | Treatment |
|---|---|---|
| 1 | FactorVAE test predictions use the labels: it scores the posterior reconstruction, whose factors are built from the true returns; the leak-free `prediction()` is never called | fixed (`predict()`) |
| 2 | FactorVAE predictions are random samples even in eval mode | fixed: the mean is returned |
| 3 | FactorVAE label uses `seq_len` instead of `pred_len` (20-day target when L = 5) | kept, stated wherever FactorVAE is compared |
| 4 | Alpha360 CSV is lag-major, the models reshape as if feature-major: the GRU steps are not in chronological order (FinFormer also sorts the columns alphabetically) | kept; attributions on the original 360 columns, mapped to (series, lag); "temporal" readings of internal states are not valid |
| 5 | MATCC reads the market file from two different paths | fixed |
| 6 | FinFormer's label is a cross-sectional z-score, not CSRank as the README says | documentation only |
| 7 | FactorVAE leaves KMID and KLEN unnormalized (z-score skips two columns) | kept |
| 8 | MASTER, MATCC and FinFormer runs are not seeded: FinBench's seed spread is uncontrolled | adapters seed every generator |
| 9 | HIST and DiscoverPLF crash on the CPU | fixed (`x.device`), identical outputs on GPU |
| 10 | Each model is trained and selected with a different rule; other data details differ (HIST's Ctrip ticker, market-cap filtering, labels reading validation prices) | kept (Option A, file 04) |

Points 3, 4 and 10 mean that some differences between models may come from implementation choices,
not architecture: say so in the limitations and when interpreting Step 7.

## The framework: one interface for every model

`xaifin.models.base.ModelAdapter` (one adapter per model, `xaifin/models/registry.py`):
- `RunConfig`: model, universe, test year, seed, T, L, data filter, hyper-parameter overrides; it
  defines the run folder.
- the adapter builds FinBench's data for that model (`day_batches`, one `DayBatch` per day), trains
  it as FinBench does (`training_loss`, `configure_optimizer`, loop settings in `hparams`), and exposes
  the prediction as a **differentiable map [N, T, F] → [N]** (`forward`), the function XAI explains.
- `load_run(run_dir)` rebuilds any trained model from its `config.json` and checkpoint.
- Alpha360 adapters all take `x` as `[N, 1, 360]` in the CSV's column order and apply FinBench's
  reshape inside, so their attributions share one input space.

## Equivalence test with FinBench (notebook 02)

Per model, DJI, test year 2020, T = 20, L = 5, seed 42, data filter off (FinBench keeps bad prices).
FinBench's own data code, model classes, training-step and evaluation functions are loaded read-only
(`notebooks/finbench_reference.py`); both sides use the same day order and random state.

Results:
- **identical initial weights** for all six models under the same seed;
- after one full epoch (982–1,006 training days), weights bit-identical for FactorVAE, HIST,
  DiscoverPLF, FinFormer; within 2e-6 (MASTER) and 4e-5 (MATCC: float32 rounding on parameters with
  near-zero gradient, moved by Adam's first steps; the adapter alone is deterministic);
- **test predictions within 2e-6**; IC, RankIC and MSE agree to the reported digits;
- FactorVAE: training bit-identical; the readout fix is the only difference. FinBench's own `test()`
  gives different RankICs for the same weights on two calls (−0.0078, −0.0093), because it samples;
- every adapter returns `[N]` on a `DayBatch` and gives finite input gradients in eval mode;
- FinFormer's MSE is not comparable (its −CCC loss ignores scale): compare it on IC and RankIC.

## For the thesis

- Table of the six models (task, inputs, normalization, loss, cross-sectional, built-in explanation).
- Table of FinBench issues and how each is treated.
- One paragraph (or appendix table) on the equivalence test: what was compared, tolerances, results.
- Pass criteria used: initial weights identical; predictions within 1e-4; RankIC within 1e-4;
  weights after one epoch within 1e-3.
