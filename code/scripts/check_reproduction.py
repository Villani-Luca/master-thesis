"""Do the retrained runs reproduce the originals? (PIANO.md, 2026-11-04)

    python code/scripts/check_reproduction.py

Compares every run of results/legacy/pre_grid_rtx5090/ (the 117 runs trained before the H200 grid) with the
run retrained at the same path under results/Regression/: test metrics, predictions and the training
history (every column but the time per epoch). Prints one line per run that differs or is not retrained
yet, and a summary.
"""

import json
from pathlib import Path

import pandas as pd

from xaifin.config import RESULTS_ROOT

LEGACY = RESULTS_ROOT / "legacy" / "pre_grid_rtx5090"


def compare(old: Path, new: Path) -> str | None:
    """None if the two runs are identical, else what differs."""
    if not (new / "metrics.json").exists():
        return "not retrained yet"
    m_old, m_new = json.loads((old / "metrics.json").read_text()), json.loads((new / "metrics.json").read_text())
    if m_old != m_new:
        keys = [k for k in m_old if m_old.get(k) != m_new.get(k)]
        return f"metrics differ: {', '.join(keys)}"
    p_old, p_new = pd.read_csv(old / "predictions.csv.gz"), pd.read_csv(new / "predictions.csv.gz")
    if len(p_old) != len(p_new) or (p_old["score"] - p_new["score"]).abs().max() > 0:
        return f"predictions differ (max |diff| {(p_old['score'] - p_new['score']).abs().max():.2e})"
    h_old, h_new = (pd.read_csv(d / "train_history.csv").drop(columns="seconds", errors="ignore") for d in (old, new))
    if not h_old.equals(h_new):
        return "training history differs"
    return None


def main() -> None:
    runs = sorted(p.parent for p in LEGACY.glob("*/*/*/*/*/metrics.json"))
    results = {run: compare(run, RESULTS_ROOT / "Regression" / run.relative_to(LEGACY)) for run in runs}
    for run, problem in results.items():
        if problem:
            print(f"{run.relative_to(LEGACY).as_posix()}: {problem}")
    identical = sum(p is None for p in results.values())
    pending = sum(p == "not retrained yet" for p in results.values())
    print(f"{identical}/{len(runs)} identical, {pending} not retrained yet, {len(runs) - identical - pending} different")


if __name__ == "__main__":
    main()
