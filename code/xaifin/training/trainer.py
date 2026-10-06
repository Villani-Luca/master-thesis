"""Generic training and evaluation of any ModelAdapter (THESIS_GUIDE.md Steps 2-3).

The model-specific parts (loss, optimizer, scheduler, gradient clipping) come from the adapter, so
one loop trains all six models as FinBench does. Step 3 adds the epoch loop around train_epoch:
model selection, early stopping, checkpoints.
"""

from typing import Iterable

import numpy as np
import torch

from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter
from xaifin.training.metrics import summary


def train_epoch(adapter: ModelAdapter, optimizer: torch.optim.Optimizer, scheduler=None, epoch: int = 0,
                days: Iterable[DayBatch] | None = None) -> float:
    """One pass over the training days (default: the train split, shuffled); returns the mean loss.

    Per day: adapter.training_loss, backward, gradient clipping by value if hparams['grad_clip'],
    optimizer step, and the scheduler step if hparams['scheduler_step'] is 'batch'. The scheduler
    steps once at the end if it is 'epoch'.
    """
    model, h = adapter.model, adapter.hparams
    model.train()
    losses = []
    for batch in days if days is not None else adapter.day_batches("train", shuffle=True):
        loss = adapter.training_loss(batch, epoch)
        optimizer.zero_grad()
        loss.backward()
        if h.get("grad_clip"):
            torch.nn.utils.clip_grad_value_(model.parameters(), h["grad_clip"])
        optimizer.step()
        if scheduler is not None and h.get("scheduler_step") == "batch":
            scheduler.step()
        losses.append(loss.item())
    if scheduler is not None and h.get("scheduler_step") == "epoch":
        scheduler.step()
    return float(np.mean(losses))


def evaluate(adapter: ModelAdapter, split: str) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    """Predictions and targets of every day of a split (eval mode), and their metrics.summary."""
    preds, labels = [], []
    for batch in adapter.day_batches(split):
        preds.append(adapter.predict(batch))
        labels.append(adapter.target(batch).numpy())
    return preds, labels, summary(preds, labels)
