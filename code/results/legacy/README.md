# Legacy runs

The first MASTER runs (seed 42, test year 2020; dji and nasdaq100), trained by the old `master_model.ipynb` (git commit `7fcec03`) and migrated to the results layout on 2026-09-29. They predate the package: no data-quality filter (`clean=False`), and a different rule for the kept epoch (best validation RankIC, patience 10) from FinBench's (stop when the training loss falls below 0.95, keep the last weights), which Step 3 uses. Moved here on 2026-10-06 so that Step 3's runs can take their place in `Regression/`; kept for reference. `xaifin.models.registry.load_run` can still open them.

## pre_grid_rtx5090/

The 117 runs trained on the RTX 5090 before the extended grid (stages 1-4: seed 42 of every model at T = 20, L = 5 on dji, nasdaq100 and sx5e, plus MASTER's seeds 0 and 5 and a few FactorVAE and MATCC ones). Only their CSV copies (config.json, metrics.json, train_history.csv, predictions.csv.gz) reached this machine, without model.pth or the results pickle, so on 2026-11-04 they were moved here and retrained on the same GPU with the same code and torch (2.11.0+cu128) to get their checkpoints back. Kept to check that the retrained runs reproduce them.
