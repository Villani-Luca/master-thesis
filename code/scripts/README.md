# scripts

Command-line entry points for batch runs over the grid of models × universes × years × seeds. They only parse arguments and call `xaifin`; no logic lives here.

Planned (see `THESIS_GUIDE.md`):

| Script | Step | Does |
|---|---|---|
| `train_rolling.py` | 3 | train a model on the rolling windows, save to `code/results/` |
| `run_portfolio.py` | 4 | long-only top-k portfolios from saved predictions |
| `grid.py` | 3 | the full training grid: what is done, half-trained or missing; trains what is left, resumable |
| `export_results.py` | 3–4 | CSV copies of runs and portfolios for git, checkpoints as one zip |
| `explain.py` | 5 | compute and save attributions |
| `analyze.py` | 7–8 | cross-model / cross-period comparisons |
