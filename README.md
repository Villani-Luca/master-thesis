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
    ├── tests/            pytest
    ├── results/          trained models, predictions, attributions
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

The datasets go in `code/data/<universe>/` and FinBench in `code/finbench/` (`git clone https://github.com/softlab-unimore/finbench code/finbench`). Then run `code/notebooks/00_environment_check.ipynb` to check the setup.

## Running

```bash
streamlit run code/app/Home.py    # GUI
cd code && python -m pytest       # tests
```
