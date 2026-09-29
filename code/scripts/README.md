# scripts

Command-line entry points for batch runs over the grid of models × universes × years × seeds. They only parse arguments and call `xaifin`; no logic lives here.

Planned (see `THESIS_GUIDE.md`):

| Script | Step | Does |
|---|---|---|
| `train_rolling.py` | 3 | train a model on the rolling windows, save to `code/results/` |
| `run_portfolio.py` | 4 | long-only top-k portfolios from saved predictions |
| `explain.py` | 5 | compute and save attributions |
| `analyze.py` | 7–8 | cross-model / cross-period comparisons |
