# Thesis Implementation Guide: Applying Explainability Methods to Financial Models

This is the working plan for turning [`docs/papers/Outline Tesi.pdf`](docs/papers/Outline%20Tesi.pdf) into code, notebooks, a reusable framework and the final graphical application. Each step says **why** it exists (which part of the outline it serves), **what to build** (notebook, package module, app page), **what it produces** (files), and **when it counts as done**.

> **Research question.** Stock-prediction models have weak predictive power (accuracy ≈ 50%, IC ≈ 0.0x). That instability carries over into long-only top-k portfolios. *How do input features drive predictions, how does that turn into allocation decisions, and does the instability come from the **data** or from the **model design**?*

---

## 0. How to use this guide

- Work through the steps in order. Every step has three layers:
  1. **Notebook** (`code/notebooks/NN_*.ipynb`): where you explore, explain and produce the thesis figures. Notebooks stay thin: they import logic, run it and plot the results.
  2. **Package** (`code/xaifin/`): the reusable framework. Logic moves here once it works in the notebook, and the notebook then imports it back.
  3. **Application** (`code/app/`): a Streamlit page per step that reads the same package and the same results store. The app contains **no logic of its own**.
- Tick the checkboxes (`- [x]`) as you go. The tracker in §4 gives the overall status.
- Each step lists the **thesis artefacts** (figures and tables) it should produce, so the writing in `latex-thesis/` can happen alongside the code.

### Mapping outline → steps

| Outline item | Guide step(s) |
|---|---|
| *Prima parte* 1–3: read the survey, the model papers and the XAI literature | Step 0.2 |
| **Step 1**: comparable model groups | Step 1 |
| **Step 2**: train with FinBench → checkpoints | Steps 2, 3 |
| **Step 2**: SHAP/LIME/time-series XAI → rank and compare features | Steps 5, 6, 7 |
| **Step 3**: qualitative analysis, data vs model | Step 7 |
| *Prima parte* 4: framework that applies XAI and saves results and metrics | Steps 2, 5, 6 |
| *Prima parte* 5: rolling-window tests, volatile periods | Steps 3, 8 |
| Allocation (long-only top-k) | Steps 4, 9 |
| **Step 4 (optional)**: counterfactual stability of the portfolio | Step 9 |
| *Seconda parte*: data-modification sensitivity tests | Step 10 |
| **Conclusion**: GUI to load models, show attributions and compare architectures | Step 11 (built incrementally from Step 1) |

---

## 1. The pipeline at a glance

```
OHLCV  ──►  Alpha158 / Alpha360 features  ──►  robust z-score (fit on train)  ──►  model (MASTER, HIST, …)
                                                                                     │
                     ┌──────────────────── predictions ŷ[date, ticker] ◄──────────────┘
                     │
      ┌──────────────┼──────────────────────────┬──────────────────────────────┐
      ▼              ▼                          ▼                              ▼
 metrics (IC,   long-only top-k            attributions A[date,ticker,T,F]   counterfactuals
 RankIC, MSE)   portfolio (CAGR, Sharpe,   (SHAP, IG, LIME, attention, …)    (how much must x or ŷ
                turnover)                   │                                 change to flip top-k?)
                                            ▼
                         rank features ─► compare across models / seeds / years / universes
                                            ▼
                               diagnosis: data-limited signal vs model design
```

---

## 2. Target architecture and conventions

### 2.1 Repository layout

The skeleton exists (2026-09-29): every folder and `xaifin` subpackage below is in place, and each `__init__.py` lists its planned modules. The modules themselves are created step by step.

```
master-thesis/
├── README.md                   # overview + setup
├── THESIS_GUIDE.md             # this plan
├── docs/
│   ├── papers/                 # the papers + Outline Tesi.pdf
│   ├── model_groups.md         # FinBench models grouped by input data
│   └── model_notes.md          # per-model notes + FinBench issues
├── latex-thesis/               # the thesis
└── code/                       # everything below
```

```
code/
├── pyproject.toml              # makes `xaifin` installable: pip install -e code
├── requirements.txt
├── data/                       # local datasets (not in git)
├── finbench/                   # FinBench clone: read-only reference (not in git)
├── tests/                      # pytest
├── xaifin/                     # the framework ("Prima parte" item 4)
│   ├── config.py               # paths, universes, model groups, rolling windows, seeds, k
│   ├── data/
│   │   ├── loading.py          # read alpha158/360, market, constituents, labels
│   │   ├── normalization.py    # RobustZScore (fit on train only), cross-sectional z-score
│   │   ├── datasets.py         # daily cross-sectional datasets -> DayBatch
│   │   ├── features.py         # feature names + feature FAMILY mapping (Alpha158/360/market)
│   │   ├── quality.py          # samples with unusable prices, discarded when config.CLEAN_DATA
│   │   └── summary.py          # Step 1 data summaries, cached in results/data_summary/
│   ├── models/
│   │   ├── base.py             # ModelAdapter protocol (see 2.3)
│   │   ├── master.py  matcc.py  factorvae.py        # Alpha158 group
│   │   ├── hist.py  discoverplf.py  finformer.py     # Alpha360 group
│   │   ├── lgbm_reference.py   # optional tree baseline with exact TreeSHAP
│   │   └── registry.py         # "MASTER" -> adapter class
│   ├── training/
│   │   ├── trainer.py          # generic loop, early stopping on valid RankIC
│   │   ├── metrics.py          # IC, RankIC, ICIR, MSE, MAE, R2 (scipy-free fallbacks)
│   │   └── rolling.py          # rolling-window orchestration
│   ├── portfolio/
│   │   ├── topk.py             # long-only top-k holdings from predictions
│   │   ├── backtest.py         # returns, CAGR, Sharpe, MaxDD, turnover
│   │   └── counterfactual.py   # Step 9
│   ├── xai/
│   │   ├── base.py             # Explainer protocol + masking/baseline utilities
│   │   ├── gradient.py         # Saliency, IntegratedGradients, GradientSHAP, DeepLIFT (Captum)
│   │   ├── shap_lime.py        # KernelSHAP / PermutationSHAP / LIME on feature GROUPS
│   │   ├── permutation.py      # cross-sectional permutation importance on IC / top-k
│   │   ├── temporal.py         # time-step / lag-window occlusion, TimeSHAP-style
│   │   ├── intrinsic.py        # attention weights (MASTER/MATCC/FinFormer), HIST concepts
│   │   ├── aggregate.py        # per-feature / per-lag / per-family global importance
│   │   └── evaluation.py       # faithfulness, stability, sanity checks (Step 6)
│   ├── analysis/
│   │   ├── agreement.py        # Spearman, Kendall, top-k Jaccard, RBO
│   │   ├── variance.py         # seed vs model vs year vs universe decomposition
│   │   └── regimes.py          # volatility regimes from market index data
│   ├── perturb/                # Step 10: OHLCV noise, ablations, shifts
│   ├── store.py                # read/write the results store (2.2); single source of truth
│   └── viz.py                  # plotly figures shared by notebooks and app
├── scripts/                    # CLI entry points for batch runs (grid over models/years/seeds)
│   ├── train_rolling.py
│   ├── run_portfolio.py
│   ├── explain.py
│   └── analyze.py
├── notebooks/                  # one notebook per step (see each step)
├── app/                        # Streamlit GUI (Step 11)
│   ├── Home.py
│   └── pages/
└── results/                    # results store (2.2)
```

### 2.2 Results store layout (one layout, read by notebooks, scripts and the app)

```
code/results/
├── Regression/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<YEAR>/
│   ├── model.pth                  # best-valid checkpoint
│   ├── config.json                # full run config (dates, hparams, feature list, git hash)
│   ├── metrics.json               # test metrics
│   ├── train_history.csv
│   └── results_sl<T>_pl<L>.pkl    # FinBench-compatible: preds, labels, pred_date, last_date, tickers
├── portfolio/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/top<k>/
│   ├── holdings.parquet           # date, ticker, weight, score, rank
│   ├── returns.parquet
│   └── metrics.json               # CAGR, Sharpe, MaxDD, Vol, turnover
├── xai/<METHOD>/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<YEAR>/
│   ├── attributions.npz           # attr[D,N,T,F] float16 on REBALANCE dates + dates, tickers, feature_names
│   ├── global.parquet             # aggregated importance per feature / lag / family
│   ├── params.json                # method, baseline, n_samples, target (pred or top-k margin)
│   └── xai_metrics.json           # faithfulness / stability scores (Step 6)
└── analysis/…                     # cross-model tables, figures exported for the thesis
```

> ✅ **Migrated 2026-09-29.** The first MASTER runs (seed 42, y2020, from the old `master_model.ipynb`) now follow this layout for dji, nasdaq100 and sp500. They have no `config.json`. The sp500 run has only `model.pth`: its test evaluation never completed.

`.gitignore` excludes `code/results/xai/**/attributions.npz` (large files). `global.parquet` and the metrics stay under version control.

### 2.3 The key abstraction: `ModelAdapter`

Every model, whatever its architecture, is wrapped so that the XAI, portfolio and app code sees the same interface:

```python
@dataclass
class DayBatch:
    date: pd.Timestamp            # prediction ("last input") date
    tickers: list[str]            # N stocks in the cross-section that day
    x: torch.Tensor               # [N, T, F] normalized features (model's own feature space)
    y: torch.Tensor               # [N] label (forward L-day return, as the model was trained)
    extras: dict                  # fixed non-feature inputs: market_value, stock2concept, …

class ModelAdapter(Protocol):
    name: str                     # "MASTER"
    group: str                    # "alpha158" | "alpha360"
    feature_names: list[str]      # length F (for MASTER/MATCC: 157 alpha + market gate features: 63 US, 42 EU)
    seq_len: int                  # T
    def build(self, cfg) -> None
    def load(self, run_dir: Path) -> None
    def day_batches(self, split: str) -> Iterator[DayBatch]
    def forward(self, x: torch.Tensor, extras: dict) -> torch.Tensor   # [N,T,F] -> [N], differentiable, eval mode
```

As implemented (`xaifin/models/base.py`): the adapter is built from a `RunConfig` (model, universe, test year, seed, T, L, `clean`, hyper-parameter overrides), which replaces `build(cfg)`. It also carries how FinBench trains the model, so the Step 3 trainer stays generic: `training_loss(batch)`, `configure_optimizer()`, `target(batch)` (the labels the predictions are scored against) and the loop settings in `hparams` (`n_epochs`, `grad_clip`, `scheduler_step`). `DayBatch.y` is the label as the model's training loss receives it (raw return for MASTER/MATCC/HIST/DiscoverPLF, daily z-score for FactorVAE and FinFormer). Each adapter implements `model_forward`; the base `forward` wraps it and turns cuDNN off when gradients are on, because cuDNN's GRU can't backpropagate in eval mode (predictions then differ by up to ~1e-4 from the cuDNN ones). Alpha360 adapters take `x = [N, 1, 360]` in the CSV's column order and `extras`: `market_value` and `stock2concept` (HIST, DiscoverPLF), `adjacency` (FinFormer).

Points to keep in mind:
- **Cross-sectional models.** MASTER, MATCC, FactorVAE, HIST and FinFormer mix information *across stocks* on the same day, so stock *i*'s prediction depends on the other stocks' inputs. Always explain a **whole day** (`x` = `[N,T,F]`) and attribute stock *i*'s output to *its own* inputs by default, holding the others fixed. Cross-stock attribution (how much stock *j* influences *i*) is an optional extra.
- **Non-feature inputs** (`extras`): HIST and DiscoverPLF also consume `<universe>_inc_matrix.npz` (stock→concept) and `<universe>_market_cap.csv`. Hold these fixed during XAI and document it.
- **Market gate features** (MASTER/MATCC, from `<nation>_market.csv`) are identical for all stocks on a date. Treat them as their own `market` family.
- **Baseline for masking.** After the robust z-score, `0` means "training median", which makes it a natural and defensible baseline. Also test a per-day cross-sectional mean baseline (Step 6).

### 2.4 Conventions

- `code/finbench/` is a **read-only reference** (it is gitignored). Copy code out of it, never import from it or edit it. This is the convention already used in [`master_model.ipynb`](code/notebooks/master_model.ipynb).
- All paths come from `xaifin.config`. Nothing is hard-coded in notebooks.
- **Data-quality filter** (decided 2026-09-29): samples whose input or label reads a ticker of the wrong company, a stale price, a price below 0.01 or a one-day jump beyond ×3 are discarded in training, validation and test (`xaifin/data/quality.py`, `config.CLEAN_DATA = True`). It removes 0.07–0.9% of the samples (2.3% for DJI, all of them `PRG.US`). FinBench keeps them, so the filter is turned off only to reproduce FinBench in the Step 2 equivalence test.
- Every run writes `config.json` with seed, dates, hyper-parameters and `git rev-parse HEAD`.
- Seeds: `0, 5, 42`, the same three as the FinBench paper (§4.1). Three seeds are needed to separate seed noise from model differences in Step 7.
- Default horizon: **T=20, L=5** (FinBench's "weekly" configuration). Alpha360 models use an effective lookback of **T=1**: one 360-vector per day, which already covers 60 days × 6 series. Optional extras: (T=5, L=1) and (T=60, L=20).
- Universes (as the outline suggests): **DJI, NASDAQ100, SX5E**. SP500/SXXP are optional scale-up.

### 2.5 Rolling windows (FinBench protocol)

| Test year | Train | Valid | Test |
|---|---|---|---|
| y2020 | 2015–2018 | 2019 | 2020 |
| y2021 | 2016–2019 | 2020 | 2021 |
| y2022 | 2017–2020 | 2021 | 2022 |
| y2023 | 2018–2021 | 2022 | 2023 |
| y2024 | 2019–2022 | 2023 | 2024 |

(The data runs into Jan 2026, so a y2025 window is possible as an extension.)

---

## Step 0: Environment and literature

### 0.1 Environment (blocker for everything that uses SHAP)

**Current state (checked 2026-09-29):** in `.venv` (Python 3.12), `torch 2.11 + cu128` works with CUDA. The scipy "Application Control policy" block is resolved: `scipy 1.18`, `scikit-learn`, `shap 0.52`, `captum 0.9`, `lime`, `streamlit 1.64` and `pyarrow 25` all import, and a KernelSHAP / Integrated Gradients smoke test passes. `xaifin` is installed in editable mode.

- [x] Fix the scipy block. Options, from least to most invasive:
  - reinstall scipy (`pip install --force-reinstall scipy`); a different wheel is sometimes not flagged;
  - use a fresh env from conda-forge (differently built binaries);
  - run the project in **WSL2** or on the RTX-5090 GPU machine mentioned in `requirements.txt`;
  - turn off Smart App Control. ⚠️ On Windows 11 it cannot be turned back on without resetting Windows, so decide this deliberately.
- [x] Add to `code/requirements.txt`: `shap`, `captum`, `lime`, `streamlit`, `pyarrow`, and optionally `timeshap`.
- [x] Create `code/pyproject.toml` and run `pip install -e code` so that `import xaifin` works from notebooks, scripts and the app.
- [x] **Notebook `00_environment_check.ipynb`**: imports, CUDA, versions, data files, and `captum`/`shap`/`lime` smoke tests on a tiny MLP (23 checks, all passing).

**Done when:** `00_environment_check.ipynb` runs top to bottom with no errors.

### 0.2 Reading (*Prima parte* 1–3)

The PDFs in `docs/papers/`:

| File | Paper | What to extract |
|---|---|---|
| `1-s2.0-S095741741630029X-main.pdf` | CI & Financial Markets survey (ESWA 2016) | the classic pipeline: data → features → model → trading |
| `13208-AAAI24.LiT.pdf` | MASTER (AAAI 2024) | task, label, market gate, T/S attention |
| `3627673.3679715.pdf` | MATCC (CIKM 2024) | differences vs MASTER |
| `12027-DuanY.pdf` | FactorVAE (AAAI 2022) | prior/posterior factors, which output to explain |
| `2110.13716v2.pdf` | HIST | concept modules, stock2concept matrix |
| `Discovering_Predictable_Latent_Factors_….pdf` | DiscoverPLF (TKDE 2023) | latent factors, extra losses |
| `Finformer_….pdf` | FinFormer (BigData 2023) | static-dynamic graph, CCC loss |
| `NIPS-2017-a-unified-approach-….pdf` | SHAP | Shapley axioms, KernelSHAP, DeepSHAP |
| `2939672.2939778.pdf` | LIME (KDD 2016) | local surrogate, sampling |
| `Explainable_AI_in_Portfolio_Management_….pdf` | XAI in PM review | where XAI is used in allocation |
| `1-s2.0-S2405844024161269-main.pdf` | Explainable DL for stock trend | a worked finance XAI example |
| *(missing)* | Qlib (arXiv 2009.11189) | origin of Alpha158/360. The reference implementation is in `code/finbench/Evaluation/features/alpha158.py` and `alpha360.py` |

- [x] For each model, write a half-page note: **what is predicted** (label = forward L-day return on adj close), **inputs**, **normalizations** (see the FinBench README table), **loss**, **whether it is cross-sectional**, and **which internal signals could be inspected** (attention, factors, concepts). → [`docs/model_notes.md`](docs/model_notes.md), which also lists **6 FinBench implementation issues** to resolve before Step 2.
- [x] XAI literature search, focused on **time-series** and, if any exist, **stock-trend** specific methods. Candidates to check: Integrated Gradients, DeepLIFT/DeepSHAP, TimeSHAP, Temporal Saliency Rescaling, Dynamask, FIT, WinIT, the "attention is (not) explanation" debate, Rashomon-set / explanation-disagreement papers, and counterfactual methods (Wachter et al., DiCE). Save the notes to `latex-thesis/chapters/02_second_chapter.tex` (related work) and to `bibliography.bib`. → Chapter 2 drafted ("Background and Related Work"); 46 references, all verified.

**Thesis artefacts:** related-work chapter; one table of the six models (task, inputs, normalization, loss, cross-sectional yes/no).

---

## Step 1: Comparable model groups and universes (Outline Step 1)

**Why:** explanations can only be compared between models that see the same data under the same protocol.

**Already done:** [`docs/model_groups.md`](docs/model_groups.md) groups all FinBench models into five groups. The core of the thesis is the two alpha groups, which are exactly the regression models listed in the outline:

| Group | Feature space | Models | Input per stock |
|---|---|---|---|
| **A: Alpha158** | 157 engineered factors (Qlib's 158 minus `VWAP0`; FinBench has no VWAP input) + market gate features for MASTER/MATCC (63 US, 42 EU) | MASTER, MATCC, FactorVAE | `[T=20, F=157(+63 US / +42 EU)]` |
| **B: Alpha360** | 6 raw normalized series (CLOSE, OPEN, HIGH, LOW, VOLUME, VWAP) × 60 lags | HIST, DiscoverPLF, FinFormer | `[60, 6]` |
| *(opt.) Reference* | Alpha158, tabular (last day) | LightGBM | `[F=157]`, with exact TreeSHAP |

Within a group, compare features one to one. Across groups, compare at the **family** level (Step 5.4).

### Tasks
- [x] `xaifin/config.py`: `UNIVERSES`, `NATION` map (dji/nasdaq100→us, sx5e→eu), `MODEL_GROUPS`, `ROLLING_WINDOWS`, `SEEDS`, `TOP_K` (e.g. 10, and 5 for DJI's ~30 stocks), `DATA_ROOT`, `RESULTS_ROOT`. → Also `CORE_UNIVERSES`, `SL_PL_CONFIGS`, `rolling_window(year)`.
- [x] `xaifin/data/features.py`: feature names plus a **family mapping**. → Done as `catalog(universe)`: family (8), source (5) and horizon (3) per feature; names checked against the CSV headers of all 5 universes. Alpha360 overlaps Alpha158 only on momentum, volume and candle, so compare the groups by source and horizon.
  - Alpha158: `kbar` (KMID, KLEN, KUP, KLOW, KSFT…), `price` (OPEN0, HIGH0, LOW0), and each rolling operator (ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP/N/D, SUMP/N/D, VMA, VSTD, WVMA, VSUMP/N/D) × window {5,10,20,30,60}.
  - Higher-level **semantic families**, shared across groups: `trend/momentum`, `volatility/range`, `volume`, `price-volume correlation`, `market`. Also horizon buckets `short (≤5d)`, `medium (10–20d)`, `long (30–60d)`.
  - Alpha360: series × lag bucket (lag 0–4 short, 5–19 medium, 20–59 long).
- [x] **Notebook `01_universes_and_data.ipynb`**: for each universe, show tickers over time (constituents), date coverage, NaN rates per feature, label distribution per year, correlation clusters among the Alpha158 features (many are near-duplicates, which matters for SHAP and for counterfactuals), and index realized volatility per year (input for Step 8). → Done for all five universes, with the thesis tables of groups and universes and a data-quality section (extreme labels, tickers mapped to the wrong company). Summaries are cached by `xaifin.data.summary` in `results/data_summary/`, charts are in `xaifin/viz.py`. Key findings in the notebook's *Observations*: the EU membership history starts on 2020-09-30 (so there is no 2020 test year for SX5E/SXXP); DJI's "Procter & Gamble" (`PRG.US`) is PROG Holdings in every test year; wrong tickers and bad prices in SXXP and SP500, now discarded by the data-quality filter (§2.4); 20 exactly redundant Alpha158 features (SUMP/N/D, VSUMP/N/D).
- [x] **App page `1_Data.py`**: pick a universe → coverage chart, feature table with family, correlation heatmap, volatility timeline. → Tabs: Coverage, Features, Labels, Redundancy, Market volatility, each with its table. Uncached universes get a "Compute summary" button.

**Done when:** the config and family mapping exist, the notebook renders every universe, and the app page works.
**Thesis artefacts:** table of groups; table of universes (N stocks, period, noise/volatility); feature-correlation figure.

---

## Step 2: Unified framework: data and model adapters

**Why:** training, XAI and the GUI all need to load any model and run it on any day through one interface (§2.3).

### Tasks
- [x] Move the MASTER code from `master_model.ipynb` into the package. The notebook was deleted from the working tree; read it with `git show 7fcec03:code/notebooks/master_model.ipynb`. Targets: `data/loading.py` (constituent filter, market merge, `extract_labels`; the market path is already done), `data/normalization.py` (`RobustZScoreNormalization`), `data/datasets.py` (the daily dataset that yields `DayBatch`; it drops the samples of `quality.discarded_samples` after building the input sequences when `config.CLEAN_DATA`), `models/master.py`, and `training/metrics.py` (the scipy-free IC/RankIC already written there). → Done 2026-10-06. Checked on dji and nasdaq100 y2020 (sl20/pl5, `CLEAN_DATA = False`): the train/valid/test samples, tensors and dates are identical to FinBench's `CSVDataset`, and the stored seed-42 checkpoints reproduce their test predictions (max diff 1e-5, GPU vs CPU) and metrics. `CLEAN_DATA` drops 3.5% of the dji y2020 samples (`PRG.US`, 1 of 29 tickers) and none of nasdaq100's. The training loop itself moves to `training/trainer.py` in Step 3.
- [x] Write an adapter for each of the other five models, copying from `code/finbench/Regression/<MODEL>/` (`train.py`, model files, `dataloader`/`load_dataset`). Keep FinBench's preprocessing for each model exactly (see the normalization column of the FinBench README table). Record every deviation in the adapter docstring.
  - FinBench issues ([`docs/model_notes.md`](docs/model_notes.md)), decided 2026-09-29: **train exactly as FinBench, fix only readout and paths.** Fixed: FactorVAE leakage and random output (1, 2), MATCC market path (5). Kept and documented: FactorVAE `seq_len` label (3), Alpha360 layout (4). Still to do: tell the supervisor, together with the data issues of notebook 01 (*Observations* 3-6) and the data-quality filter.
  - [x] FactorVAE: model code + leak-free, deterministic `predict()` in `xaifin/models/factorvae.py` (tests in `code/tests/`). The adapter wrapper comes with the rest of Step 2.
  - [x] **Alpha158 group** (2026-10-06): `models/base.py` (`RunConfig`, `ModelAdapter`), `MASTERAdapter`, `MATCCAdapter` (+ `models/matcc.py`, `training/lr_scheduler.py`), `FactorVAEAdapter` (+ `data.datasets.factorvae_splits`, Qlib's calendar windows with filled gaps). Checked on dji y2020: FactorVAE's train/valid/test samples identical to FinBench's `TSDataSampler`; MATCC's outputs and learning-rate schedule identical to FinBench's; the MASTER checkpoint gives its stored test metrics through the adapter; 30 training days run for each adapter. Two new FinBench findings: issues 7 (FactorVAE leaves KMID/KLEN unnormalized) and 8 (MASTER/MATCC runs not seeded) in `model_notes.md`.
  - [x] **Alpha360 group** (2026-10-06): `HISTAdapter`, `DiscoverPLFAdapter`, `FinFormerAdapter` (+ `models/hist.py`, `discoverplf.py`, `finformer.py`; `data.datasets.RowDataset`, `hist_splits`, `finformer_splits`). x is `[N, 1, 360]` in the CSV's column order for all three; each adapter applies FinBench's own reshape inside, so their attributions share one input space. Checked on dji y2020: train/valid/test samples, market values, concept rows and FinFormer's graph identical to FinBench's loaders; outputs identical to FinBench's model classes with the same weights; `concordance_cc` identical to audtorch's. All six adapters train for 30 days on the GPU and give finite input gradients in eval mode. New findings: issues 9 (HIST/DiscoverPLF only run on a GPU, fixed) and 10 (training-loop differences), and issue 8 extends to FinFormer.
  - HIST/DiscoverPLF: pass `stock2concept` and `market_value` through `extras`.
  - FinFormer: pass the sector/industry adjacency through `extras`. Its label is a daily cross-sectional z-score, not the CSRank the README describes.
  - Alpha360 models: compute attributions on the **original 360 columns**, then map them to (series, lag).
- [ ] `models/registry.py`: `get_adapter("HIST")`.
- [x] **Equivalence test:** for each model, train briefly on `dji` y2020 with the FinBench script *and* with the adapter using the same seed, then check that predictions and metrics match within tolerance (except where an issue from `model_notes.md` was deliberately fixed). Run it with `CLEAN_DATA = False`, since FinBench keeps the bad prices. → Done 2026-10-06 in **notebook `02_model_adapters.ipynb`** with `notebooks/finbench_reference.py` (FinBench's own data code, models, training-step and evaluation functions, loaded read-only; the scripts themselves can't run here) and the first part of `training/trainer.py` (`train_epoch`, `evaluate`). One full epoch, same initial weights, day order and random state on both sides: all six pass (initial weights identical; weights after one epoch identical or within 4e-5, Adam on rounding-level gradients; test predictions within 2e-6; FactorVAE against FinBench's leak-free `prediction()`). One section per model.
- [ ] **App page `2_Models.py`**: pick model / universe / year / seed → load the checkpoint, show config, training curve, test metrics, and the prediction-vs-label scatter for a chosen date.

**Done when:** all six adapters pass the equivalence check and `adapter.forward(x, extras)` returns `[N]` on a `DayBatch` for each model.

---

## Step 3: Rolling-window training and checkpoints (Outline Step 2, *Prima parte* 5)

### Tasks
- [ ] `training/trainer.py`: a generic loop with early stopping on **valid RankIC** (as in the current notebook), best-state checkpointing, and saving to the §2.2 layout.
- [ ] `training/rolling.py` + `scripts/train_rolling.py --model MASTER --universe dji --years 2020-2024 --seeds 0 5 42 --sl 20 --pl 5`. Skip runs that already exist (resumable).
- [x] Migrate the existing MASTER results to the new layout (done 2026-09-29, see §2.2).
- [ ] Run the grid **in stages**, checking each stage before going on:
  1. MASTER × dji × y2020 × seed 42 (already done, re-run through the package);
  2. all 6 models × dji × y2020 × seed 42 → compare with the FinBench tables (`code/finbench/results.md`). FactorVAE will score lower if its test-time leakage is fixed;
  3. all 6 models × {dji, nasdaq100, sx5e} × 2020–2024 × seed 42. SX5E only has 2021–2024: its membership history starts on 2020-09-30, so the 2020 universe is empty (notebook 01);
  4. add seeds 0 and 5. The full grid is 6 × 3 × 5 × 3 = **270 runs**, so estimate the time per run first.
- [ ] **Notebook `03_rolling_training.ipynb`**: tables of IC, RankIC, ICIR and MSE per model × universe × year (mean ± std over seeds). Sanity-check that the numbers are in the FinBench range. This notebook also documents *how weak the signal is*, which is the starting point of the thesis.
- [ ] **App page `2_Models.py`** (extended): a metrics grid (heatmap model × year) for the selected universe.

**Done when:** a checkpoint exists for every cell of the chosen grid, and the metrics summary table is saved to `results/analysis/metrics_summary.csv`.
**Thesis artefacts:** model-performance table; IC-per-year plot.

---

## Step 4: Portfolio baseline (long-only top-k)

**Why:** the outline cares about how explanations translate into **allocation**. Steps 7–9 need a reference portfolio for every run.

### Tasks
- [ ] `portfolio/topk.py`: on each **rebalance date** (every L=5 business days, as FinBench does with `freq=f'{pl}B'`), pick the top-k stocks by predicted score with equal weights. Store `score`, `rank` and the **margin to the k-boundary** (`score_i − score_(k+1)` for held stocks, `score_k − score_i` for excluded ones). Step 9 builds on this margin.
- [ ] `portfolio/backtest.py`: daily returns from `adj_close`, then CAGR, Sharpe, Sortino, MaxDD, annualized volatility and **turnover**. Use FinBench `Evaluation/portfolio/` (`transforms.create_long_short_portfolio_history`, `returns.portfolio_daily_returns`) as the reference implementation and reproduce its numbers for one run. Benchmark: the equal-weight universe and the index.
- [ ] **Seed-instability metric:** for the same model/year, the Jaccard overlap of top-k holdings across seeds and the turnover between seed portfolios. This is the first number that shows "small prediction changes → different portfolio".
- [ ] `scripts/run_portfolio.py` for the grid.
- [ ] **Notebook `04_portfolio_baseline.ipynb`**: equity curves per model, a CAGR/Sharpe table, the holdings overlap across seeds and models, and the distribution of boundary margins.
- [ ] **App page `6_Portfolio.py`** (first version): equity curves, a holdings heatmap (date × ticker), and the margin histogram.

**Done when:** `portfolio/…/metrics.json` exists for every trained run and one run matches FinBench's evaluation output.
**Thesis artefacts:** long-only top-k table; seed-overlap figure.

---

## Step 5: XAI engine: attributions (Outline Step 2, *Prima parte* 4)

**Goal:** for every (model, universe, year, seed), produce attributions of the prediction to the inputs, save them, and rank features.

### 5.1 Methods to implement (`xaifin/xai/`)

| Family | Method | Resolution | Notes |
|---|---|---|---|
| Gradient (Captum) | Saliency, **Integrated Gradients**, GradientSHAP, DeepLIFT | full `[T,F]` per stock | fast on GPU; main workhorse. Baseline = 0 (training median) |
| Shapley / surrogate | **KernelSHAP / PermutationSHAP**, **LIME** | **feature groups** | full `T×F` (e.g. 20×221) is too many players, so mask a feature across all timesteps (F players), or use families (~35 players) |
| Global, model-agnostic | **Cross-sectional permutation importance** | per feature / family | shuffle a feature *across stocks within each day*, then measure the drop in RankIC and the change in top-k membership. The metric closest to finance |
| Time-series specific | lag-window occlusion, TimeSHAP-style (feature × time-step × event) | per lag / window | answers "which days in the lookback matter?" |
| Intrinsic | attention weights (MASTER T/S attention, gate; MATCC; FinFormer), HIST concept weights, FactorVAE factor exposures | model-specific | the MASTER notebook already notes that the attention modules must be patched to *return* their weights. Use this as supporting evidence only (attention ≠ explanation) |
| Reference | TreeSHAP on LightGBM | exact | anchors "what the data allows" (Step 7) |

- [ ] `xai/base.py`: an `Explainer` protocol, `explain(adapter, batch, target) -> attr[N,T,F]`. `target` is either `"pred"` (the raw score) or `"topk_margin"` (the score minus the k-boundary score, which explains *portfolio membership* rather than the score).
- [ ] Implement the methods above, one module per family.
- [ ] Explain only on **rebalance dates** (≈50 per year) to keep storage manageable: 50 × 100 stocks × 20 × 221 × 2 bytes ≈ 45 MB per run in float16.

### 5.2 Aggregation and ranking (`xai/aggregate.py`)
- [ ] Normalize each sample: `ã = |a| / Σ|a|`, so that models with different output scales are comparable.
- [ ] Global importance per **feature** (sum over T), per **lag** (sum over F), per **family** and per **semantic family**. Keep the mean and the dispersion over dates/stocks. Also keep the *signed* mean, to see the direction of the effect.
- [ ] Rank the features for each run. Save `global.parquet`.

### 5.3 Batch runner
- [ ] `scripts/explain.py --method ig --model MASTER --universe dji --years 2020-2024 --seeds 0 5 42`.

### 5.4 Notebooks and app
- [ ] **Notebook `05_xai_single_model.ipynb`**: MASTER × dji × y2020. Run every method, then show a `T×F` heatmap for one stock/day, the top-20 features globally, a family bar chart, the lag profile, permutation importance vs IG, and attention maps. Only after this works, generalize to the other models.
- [ ] **App page `3_Explain.py`**: pick model/universe/year/seed/method → (a) global ranking bar chart plus family breakdown, (b) a date + ticker selector showing the `T×F` heatmap and a waterfall of the top features, (c) an "explain now" button that runs a gradient method on demand for one day (GPU, cached), (d) the attention viewer for models that expose it.

**Done when:** `global.parquet` exists for at least IG and permutation importance for every run in the grid, and SHAP/LIME exist for the stage-2 grid (every model, dji, y2020).
**Thesis artefacts:** top-feature tables per model; family-importance figure; example local explanation.

---

## Step 6: Evaluating the explanations

**Why:** with weak models, explanations can be noisy too. The explanations have to be trustworthy before any conclusion is drawn from them (*Prima parte* 4: "salvare risultati ed eventuali metriche di valutazione").

### Tasks (`xai/evaluation.py`, results in `xai_metrics.json`)
- [ ] **Faithfulness:** deletion/insertion curves. Mask the top-k attributed features (with the baseline), record |Δ prediction| and ΔRankIC, and compare with random masking. Report AOPC.
- [ ] **Stability / robustness:** similarity of attributions under small input noise (max-sensitivity). Also similarity between neighbouring days for the same stock.
- [ ] **Sanity check** (Adebayo et al. 2018): randomize the model weights; the attributions must change.
- [ ] **Method agreement:** rank correlation between IG, SHAP, LIME and permutation importance on the same run.
- [ ] **Baseline sensitivity:** zero baseline vs per-day cross-sectional mean.
- [ ] **Notebook `06_xai_evaluation.ipynb`**: pick the methods to trust for the rest of the thesis, and justify the choice.
- [ ] **App:** a "quality" tab on `3_Explain.py` with the deletion curve and the method-agreement heatmap.

**Done when:** a documented choice of primary method(s) with the evidence behind it.

---

## Step 7: Cross-model comparison and the data-vs-model diagnosis (Outline Steps 2 and 3)

**Goal:** rank and compare features across models, check whether some features consistently dominate, and decide whether the instability comes from the **data** or the **architecture**.

### Tasks
- [ ] `analysis/agreement.py`: Spearman ρ, Kendall τ, top-k Jaccard and **rank-biased overlap** between importance rankings.
  - Within group A (and within B): at the feature level.
  - Across groups: at the semantic-family / horizon-bucket level (§Step 1).
- [ ] `analysis/variance.py`: collect the importance vectors indexed by (model, seed, year, universe) and compare the agreement:
  - **same model, different seeds** (training noise),
  - **different models, same seed/year** (architecture),
  - **same model, different years** (non-stationarity of the data),
  - **same model, different universes** (universe effect).
  A variance decomposition (ANOVA-style) over these factors gives a single table.
- [ ] **Diagnosis logic** (write it explicitly in the thesis):
  - models **agree** on features, and IC is low for all of them → the signal in the data is limited or noisy (**data-driven instability**);
  - models **disagree** while having similar IC → several equally good but different explanations exist (Rashomon effect), so architecture drives *which* signal is used (**model-driven**);
  - seed disagreement ≈ model disagreement → the explanations themselves are not stable, so no architectural conclusion can be drawn;
  - LightGBM/TreeSHAP reference: if the deep models' top features match it, the data sets the pattern.
- [ ] Link to allocation: repeat the comparison with `target="topk_margin"` attributions. Do the features that drive the *portfolio* differ from those that drive the *score*?
- [ ] **Notebook `07_cross_model_comparison.ipynb`**: agreement heatmaps (model × model), seed-vs-model agreement boxplots, the variance-decomposition table, and "consistently dominant features" (features in the top-10 across ≥ X% of runs).
- [ ] **App page `4_Compare.py`**: multi-select models → side-by-side rankings, an agreement heatmap, a family radar/bar chart, and seed-dispersion error bars.

**Done when:** the variance-decomposition table exists and a written diagnosis is backed by it.
**Thesis artefacts:** the core results chapter: agreement heatmaps, dominant-feature table, diagnosis.

---

## Step 8: Period and regime analysis over rolling windows (*Prima parte* 5)

**Goal:** find out whether particular periods (high volatility) change model behaviour and explanations.

### Tasks
- [ ] `analysis/regimes.py`: realized volatility of the index from `<nation>_market.csv` (e.g. `std_price_change_20_*`), then label each date/month low/medium/high volatility. Mark the known events: 2020 COVID crash, 2022 rate-hike bear market.
- [ ] Per regime and per year: IC/RankIC, portfolio returns, **importance drift** (the distance between importance vectors of consecutive windows), and family shares (e.g. does volatility/market importance rise in turbulent periods?).
- [ ] A monthly (not yearly) attribution time-series inside each test year.
- [ ] **Notebook `08_period_analysis.ipynb`**: a timeline plot (index vol, IC and family shares stacked on shared x-axes), and a regime × model table.
- [ ] **App page `5_Over_time.py`**: pick model/universe → importance over time with volatility shading, plus a year-to-year drift chart.

**Done when:** a figure and a short text answering "are some periods more unstable, and do the explanations shift with them?".

---

## Step 9 (optional): Counterfactual stability of the portfolio (Outline Step 4)

**Goal:** measure how much inputs or predictions must change to alter the portfolio's composition.

### 9.1 Prediction space (cheap, do first)
- [ ] Use the margins from Step 4: the minimal |Δŷ| to flip each stock in or out of the top-k.
- [ ] Compare the margins with the **seed standard deviation of predictions** (Step 3). *Decision fragility* = the fraction of holdings whose margin is smaller than the seed noise.
- [ ] Flip rate under Gaussian noise ε on predictions, for ε in a grid; the resulting turnover and return change.

### 9.2 Input space (with realistic constraints)
- [ ] **(A) Feature space, gradient-based** (Wachter-style): minimize `‖δ‖₁ + λ·hinge(margin(x+δ))` using `adapter.forward`. Constraints: stay inside the clipped z-range [−3, 3], perturb only mutable features (not `market`), keep perturbations sparse, and move correlated features together (clusters from Step 1).
- [ ] **(B) OHLCV space, black-box** (most realistic, because the features are derived): perturb the raw price/volume path of the last few days (e.g. close ±x%), **recompute Alpha158/360** with the pipeline in `finbench/Evaluation/features/`, re-normalize, and re-predict. Bisection/random search on x finds the minimal perturbation that flips membership.
- [ ] Metrics: distance of the counterfactual (L1 in z-space, number of features changed, % price move), flip rate at a given ε, and which features the counterfactuals change most (compare with Step 5 attributions).
- [ ] `portfolio/counterfactual.py`, **notebook `09_counterfactual_stability.ipynb`**.
- [ ] **App page `6_Portfolio.py`** (extended): select date + stock → margin, the minimal counterfactual, and a "what-if" slider on price perturbation that shows the new top-k.

**Done when:** a distribution of minimal flip perturbations per model, plus a fragility comparison across models.

---

## Step 10: Data-perturbation sensitivity tests (*Seconda parte*)

These are to be defined with the supervisor based on the results. Scaffold in `xaifin/perturb/` so they plug into the same pipeline:
- [ ] Noise injection on OHLCV at several σ → recompute features → Δ IC, Δ portfolio, Δ attributions.
- [ ] Feature-family ablation: drop a family and **retrain**, a stronger test of importance than masking.
- [ ] Time shift / delayed data, missing days, and universe swap (train on one, test on another).
- [ ] **Notebook `10_perturbation_tests.ipynb`**, and an app tab on `4_Compare.py` or a separate page.

---

## Step 11: Graphical application (Conclusion of the outline)

> *"The expected outcome is a graphical interface for loading models, visualizing feature attributions, and comparing explanations across architectures."*

**Stack:** **Streamlit** multipage app (pure Python, uses `xaifin` directly, supports caching and GPU on-demand explanations). Run with `streamlit run code/app/Home.py`.

The pages are added step by step, so the app grows together with the research:

| Page | Built in step | Content |
|---|---|---|
| `Home.py` | 11 | project summary, results-store status (which runs, attributions and portfolios exist) |
| `1_Data.py` | 1 | universe coverage, features and families, correlations, volatility |
| `2_Models.py` | 2–3 | load a checkpoint, config, training curve, metrics grid |
| `3_Explain.py` | 5–6 | global and local attributions, heatmaps, on-demand IG, attention, explanation quality |
| `4_Compare.py` | 7, 10 | cross-model/seed agreement, dominant features, variance decomposition |
| `5_Over_time.py` | 8 | importance drift vs volatility regimes |
| `6_Portfolio.py` | 4, 9 | top-k holdings, margins, counterfactual what-if |

### Tasks
- [ ] `app/Home.py` plus a shared sidebar (universe, model, seed, year, method) stored in `st.session_state`.
- [ ] Use `@st.cache_resource` for loaded models and `@st.cache_data` for results-store reads.
- [ ] Every chart function lives in `xaifin` (e.g. `xaifin/viz.py`) so the notebooks and the app draw identical figures.
- [ ] "Upload/choose a checkpoint" flow: pick a run directory → the adapter comes from `config.json` → ready to explain.
- [x] Short README section on how to launch the app.

**Done when:** a new user can start the app, load any trained model, see its attributions, and compare it with another architecture without touching code.

---

## Step 12: Thesis writing hooks

| Chapter (`latex-thesis/chapters/`) | Fed by |
|---|---|
| Introduction | outline objective, the "weak signal → unstable allocation" motivation (Step 3/4 numbers) |
| Background & related work | Step 0.2 |
| Methodology: data, models, rolling protocol, XAI methods, evaluation | Steps 1, 2, 3, 5, 6 |
| Results: performance, attributions, cross-model comparison, periods, counterfactuals | Steps 3–9 |
| The application | Step 11 (screenshots) |
| Conclusions: data vs model diagnosis | Step 7 |

Export every thesis figure from the notebooks to `latex-thesis/images/results/` with fixed filenames, so re-running a notebook updates the thesis.

---

## 3. Suggested order and milestones

| Milestone | Steps | Outcome |
|---|---|---|
| **M1: Foundations** | 0, 1, 2 | environment fixed, package skeleton, 6 adapters verified |
| **M2: Checkpoints** | 3, 4 | rolling grid trained (stage 3), portfolios computed |
| **M3: Explanations** | 5, 6 | attributions for the grid + validated choice of methods |
| **M4: Analysis** | 7, 8 | cross-model diagnosis, period analysis → core results |
| **M5: Extensions** | 9, 10 | counterfactual stability, perturbation tests |
| **M6: Application & write-up** | 11, 12 | GUI complete, thesis chapters |

Build the app page for each step **in the same milestone** as the step, not at the end.

---

## 4. Progress tracker

| Step | Notebook | Package | App page | Status |
|---|---|---|---|---|
| 0 Environment & reading | `00_environment_check` | `pyproject.toml` | – | ✅ done (model notes + chapter 2 drafted; read the papers yourself too) |
| 1 Groups & universes | `01_universes_and_data` | `config`, `data/features` | `1_Data` | ✅ done |
| 2 Adapters | `02_model_adapters` | `data/*`, `models/*` | `2_Models` | ◐ data pipeline, `training/metrics`, all six adapters and the equivalence test (all pass) done; registry and app page to do |
| 3 Rolling training | `03_rolling_training` | `training/*`, `scripts/train_rolling` | `2_Models` | ◐ MASTER seed 42 y2020 on dji/nasdaq100 (sp500: checkpoint only); results migrated to the §2.2 layout |
| 4 Portfolio baseline | `04_portfolio_baseline` | `portfolio/topk`, `backtest` | `6_Portfolio` | ☐ |
| 5 XAI engine | `05_xai_single_model` | `xai/*`, `scripts/explain` | `3_Explain` | ☐ |
| 6 XAI evaluation | `06_xai_evaluation` | `xai/evaluation` | `3_Explain` (quality) | ☐ |
| 7 Cross-model diagnosis | `07_cross_model_comparison` | `analysis/agreement`, `variance` | `4_Compare` | ☐ |
| 8 Period analysis | `08_period_analysis` | `analysis/regimes` | `5_Over_time` | ☐ |
| 9 Counterfactuals (opt.) | `09_counterfactual_stability` | `portfolio/counterfactual` | `6_Portfolio` | ☐ |
| 10 Perturbation tests | `10_perturbation_tests` | `perturb/*` | `4_Compare` | ☐ |
| 11 Application | – | `viz.py` | `Home` + all pages | ◐ `Home.py` lists the trained runs; README explains how to launch it |
| 12 Thesis | – | – | – | ☐ |
