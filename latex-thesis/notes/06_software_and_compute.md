# 6. Software, reproducibility and compute

Sources: `code/xaifin/`, `code/scripts/`, `code/app/`, `THESIS_GUIDE.md` §2 and Step 11,
`code/scripts/remote/`, `decision_claude.md` (H200 measurements, 2026-10-22).

## The `xaifin` package

Three layers, so that the same logic feeds notebooks, scripts and the app: **notebooks** explore and
produce the figures; the **package** holds all logic; the **app** has no logic of its own.

| Module | Role |
|---|---|
| `config` | universes, regions, model groups, (T, L) configurations, seeds, top-k, rolling windows, data filter switch |
| `data/` | loading (constituent filter, market merge, labels), robust z-score, daily datasets (`DayBatch`), feature catalog with families, data-quality filter, cached data summaries |
| `models/` | `ModelAdapter` base class, six adapters and model code, registry (`get_adapter`, `load_run`, `find_runs`) |
| `training/` | generic loop with FinBench's selection rules, rolling runner, metrics (IC, RankIC, ICIR…), learning-rate scheduler |
| `portfolio/` | top-k holdings with margins and overlap; backtest, benchmarks, performance metrics |
| `store` | CSV exports of runs and portfolios, checkpoint archives |
| `viz` | every chart, shared by notebooks and app; fixed colorblind-safe colors per model |
| `xai/`, `analysis/` | to be written (Steps 5–8) |

**Results store** (`code/results/`): one folder per run,
`Regression/<model>/<universe>/sl<T>_pl<L>/seed<S>/y<year>/` with `model.pth`, `config.json` (all
settings plus a `training` section: rule, kept epoch, validation metrics, time, device),
`metrics.json`, `train_history.csv`, predictions; portfolios in `results/portfolio/`; summaries in
`results/analysis/`.

Scripts: `train_rolling.py` (resumable: finished runs are skipped, interrupted ones resume from the
last epoch with identical results), `grid.py` (status and scheduling of the whole grid),
`run_portfolio.py`, `export_results.py`, `compare_finbench.py`.

## Reproducibility

- Every generator (Python, NumPy, torch CPU and CUDA) is seeded and cuDNN is deterministic; the
  adapter alone gives identical weights on repeated runs.
- Equivalence with FinBench's own code verified per model (file 03).
- Resume state (`resume.pt`) holds weights, optimizer, scheduler and random generators, written
  atomically: an interrupted run gives the same result as an uninterrupted one.
- On the H200, running 2 or 6 CPU threads per process, and 1, 2 or 4 processes sharing a GPU, gave
  **bit-identical weights and predictions** (30 cases).
- FinBench's code stays read-only; every fix lives in `xaifin` and is covered by tests (`code/tests/`).

## Compute

- First grid (252 runs): one **RTX 5090**, ~88 GPU hours.
- Extended grid (1242 runs): one **H200 NVL** on an external server, 8 runs sharing the GPU, a RAM
  budget of ~128 GiB, 16 CPU cores. Measurements: one run uses the GPU at 4–34% and ≤ 2.3 GiB of
  VRAM — the models are bound by CPU/Python latency, not by the GPU; 8 concurrent runs give about 3×
  the throughput of one. Peak RAM per run: DJI 3–6 GiB, SX5E 4–10, NASDAQ-100 7–17, S&P 500 26–63,
  SXXP 27–72 (RAM, not the GPU, limits the parallelism). Time per run (one test year) on the H200:
  FactorVAE ~2.5 h, DiscoverPLF 0.5–0.9 h, MATCC 0.6–0.8 h, HIST ~0.4 h, FinFormer 0.14–0.8 h,
  MASTER ~0.1 h. Total ~39 GPU-days serial, ~13 days with concurrency.
- Worth a sentence in the thesis: these models are small; the cost is FactorVAE's sequential
  attention and the size of the cross-section, not the parameter count.

## The application (Streamlit, `streamlit run code/app/Home.py`)

Built so far:
- `Home`: project summary and results-store status.
- `1_Data`: per universe, tabs Coverage, Features (with family), Labels, Redundancy, Market volatility.
- `2_Models`: pick model / universe / year / seed → stored metrics, configuration, hyper-parameters;
  training curves (loss, validation IC/RankIC per epoch); test predictions recomputed through the
  adapter next to the stored metrics, daily RankIC, prediction vs label with the top-k highlighted;
  a metrics heatmap model × year per universe with seed spread.
- `6_Portfolio`: equity curves vs benchmarks, metrics and yearly returns, holdings map, margin
  histogram, one period's candidates.
- Checked headless with Streamlit's `AppTest` (recomputed metrics equal the stored ones).

To build (deadline scope): `3_Explain` (global ranking and family breakdown, T × F heatmap per date
and stock, attention), `4_Compare` (rankings side by side, agreement heatmap, seed dispersion).
Screenshots for the "Application" chapter once these exist.
