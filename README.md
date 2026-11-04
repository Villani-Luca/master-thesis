# Applying Explainability Methods to Financial Models

Master's thesis, University of Modena and Reggio Emilia. It studies which input features drive the predictions of recent stock-forecasting models (MASTER, MATCC, FactorVAE, HIST, DiscoverPLF, FinFormer), how those attributions translate into long-only top-k portfolio decisions, and whether prediction instability comes from the data or from the model design.

The step-by-step plan is in [`THESIS_GUIDE.md`](THESIS_GUIDE.md).

## Layout

```
├── THESIS_GUIDE.md       step-by-step plan (notebook + package + app per step)
├── docs/
│   ├── papers/           papers and the thesis outline
│   ├── model_groups.md   FinBench models grouped by input data
│   └── model_notes.md    per-model notes and known FinBench issues
├── latex-thesis/         the thesis (LaTeX, biblatex/biber)
└── code/
    ├── xaifin/           the framework: data, models, training, portfolio, xai, analysis, perturb
    ├── notebooks/        one notebook per step
    ├── scripts/          command-line batch runs
    ├── app/              Streamlit GUI
    ├── results/          trained runs, predictions, portfolios, analysis tables (see Results below)
    ├── data/             datasets (local only, not in git)
    └── finbench/         FinBench clone, read-only reference (local only, not in git)
```

## Setup

From the repository root, with Python 3.12:

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install -r code/requirements.txt
pip install -e code               # makes `import xaifin` work everywhere
```

The datasets (from the FinBench authors' shared Drive folder) go in `code/data/<universe>/` and FinBench is in `code/finbench/`: a plain copy of https://github.com/softlab-unimore/finbench at commit `71938a4` (2026-09-07), kept read-only. Then run `code/notebooks/00_environment_check.ipynb` to check the setup.

## Running

```bash
streamlit run code/app/Home.py                        # GUI
python code/scripts/train_rolling.py --help           # rolling-window training
python code/scripts/run_portfolio.py --help           # long-only top-k portfolios
python code/scripts/export_results.py --zip ckpt.zip  # CSV copies + checkpoints archive
```

## Results

The small files of `code/results/` are in git; the large ones (marked *local*) stay on disk and are kept on a separate drive:

| Folder | Content |
|---|---|
| `analysis/` | summary tables: `metrics_summary.csv` (test metrics, one row per run), `portfolio_summary.csv`, `portfolio_margins.csv`, the comparison with FinBench's tables (see its README) |
| `Regression/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<test year>/` | one trained run: `config.json` (hyper-parameters, kept epoch), `metrics.json` (test IC, RankIC, MSE, …), `train_history.csv` (one row per epoch); *local*: `predictions.csv.gz` (date, realized, ticker, score, label), `results_sl<T>_pl<L>.pkl`, `model.pth` |
| `portfolio/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/top<k>/` | one portfolio over the test years: `metrics.json` (CAGR, Sharpe, turnover, yearly returns), `returns.csv.gz` (daily returns of the portfolio, equal-weight benchmark and index); *local*: `holdings.csv.gz` (every candidate stock per rebalance, with rank and margin to the top-k), the `.parquet` files |
| `data_summary/` | Step 1 data summaries per universe |

The local files (about 6 GB for the full grid) are not in git: copy `code/results/Regression/` and `code/results/portfolio/` from the drive into `code/results/` (or unzip the checkpoints archive of `export_results.py --zip` into `code/`).
