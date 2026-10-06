# Analysis tables

- `finbench_table2.csv`: FinBench's published regression results (`code/finbench/results.md`, Table 2, `imgs/regression_results.png`), transcribed 2026-10-06: test MSE and MAE of the six models by universe and horizon L (lookbacks 5/20/60 for L = 1/5/20). Aggregated as in FinBench's protocol (paper §4.1: rolling test years 2020-2024, seeds 0, 5, 42; to be confirmed in the paper). FinBench's FactorVAE numbers come from its leaky test readout (issue 1 in `docs/model_notes.md`), and its MASTER, MATCC and FinFormer runs were not seeded (issue 8).
