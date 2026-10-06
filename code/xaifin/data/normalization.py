"""Feature and label normalizations of the FinBench models.

Source: code/finbench/Regression/MASTER/load_dataset.py (RobustZScoreNormalization) and
base_model.py (zscore, drop_extreme). FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.
"""

import numpy as np
import pandas as pd
import torch

MAD_TO_STD = 1.4826  # MAD of a normal distribution x 1.4826 = its standard deviation
CLIP = 3.0


class RobustZScore:
    """Per-feature (x - median) / (1.4826 * MAD), clipped to [-3, 3], with statistics fit on one table.

    Fit it on the training period only, then apply it to every split. After the transform, 0 is the
    training median of each feature: the natural masking baseline (THESIS_GUIDE.md §2.3).

    `columns` defaults to every column except instrument, date and Label: the same columns as
    FinBench's `df.columns[2:-1]`.
    """

    def __init__(self, train: pd.DataFrame, columns: list[str] | None = None, eps: float = 1e-12):
        self.columns = columns or [c for c in train.columns if c not in ("instrument", "date", "Label")]
        self.median = train[self.columns].median()
        self.mad = (train[self.columns] - self.median).abs().median() + eps

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """A copy of `df` with the feature columns normalized."""
        out = df.copy()
        scaled = (out[self.columns] - self.median) / (self.mad * MAD_TO_STD)
        out[self.columns] = np.clip(scaled, -CLIP, CLIP)
        return out


def cs_zscore(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional z-score of one day's labels (FinBench `zscore`, the CSZScoreNorm of Qlib)."""
    return (x - x.mean()).div(x.std())


def drop_extreme(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask of the labels without the int(2.5% * N) lowest and highest, and the kept labels.

    Training only: FinBench never drops labels in validation or test.
    """
    n_tail = int(0.025 * x.shape[0])
    if n_tail == 0:
        return torch.ones_like(x, dtype=torch.bool), x
    _, order = x.sort()
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[order[n_tail:-n_tail]] = True
    return mask, x[mask]
