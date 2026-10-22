"""Generic training and evaluation of any ModelAdapter (THESIS_GUIDE.md Steps 2-3).

The model-specific parts (loss, optimizer, scheduler, gradient clipping, the rule that picks the
final weights) come from the adapter, so one loop trains all six models as FinBench does
(decided 2026-10-06: Option A, every model keeps FinBench's own selection rule).
"""

import collections
import copy
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LRScheduler

from xaifin.data.datasets import DayBatch
from xaifin.models.base import ModelAdapter
from xaifin.training.metrics import summary

# hparams['selection'] -> what FinBench keeps (THESIS_GUIDE.md Step 3, docs/model_notes.md issue 10).
SELECTION_RULES = {
    "train_loss_threshold": "MASTER (base_model.py fit): stop after the first epoch whose training loss is at most "
                            "train_stop_loss_thred, keep the last weights",
    "last": "MATCC (train.py): train n_epochs, keep the last weights",
    "min_valid_loss": "FactorVAE (train.py main): keep the weights of the lowest validation loss (the VAE loss)",
    "smoothed_valid_ic": "HIST, DiscoverPLF (train.py main): from eval_from_epoch on, evaluate the average of the "
                         "last smooth_steps epochs' weights; keep the average with the best validation IC (daily "
                         "mean); stop after early_stop epochs without improvement",
    "pooled_valid_ic": "FinFormer (finformer.py fit): keep the weights of the best validation IC pooled over all "
                       "stock-days; stop after early_stop epochs without improvement",
}


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


@torch.no_grad()
def validation_loss(adapter: ModelAdapter, split: str, epoch: int) -> float:
    """The training loss on a split in eval mode, as FinBench's FactorVAE validate(): sum(loss x N) / days."""
    adapter.model.eval()
    total, days = 0.0, 0
    for batch in adapter.day_batches(split):
        total += adapter.training_loss(batch, epoch).item() * len(batch.tickers)
        days += 1
    return total / days


def average_params(params_list) -> dict:
    """FinBench's average_params (HIST/train.py, DiscoverPLF/train.py): the mean of several state_dicts."""
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    for params in params_list:
        for k, v in params.items():
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def _scheduler_state(scheduler) -> dict:
    """A scheduler's state, including the schedulers it wraps (MATCC's ChainedScheduler holds three)."""
    own = {k: v for k, v in scheduler.__dict__.items() if k != "optimizer" and not isinstance(v, LRScheduler)}
    inner = {k: _scheduler_state(v) for k, v in scheduler.__dict__.items() if isinstance(v, LRScheduler)}
    return {"own": own, "inner": inner}


def _load_scheduler_state(scheduler, state: dict) -> None:
    scheduler.__dict__.update(state["own"])
    for k, inner in state["inner"].items():
        _load_scheduler_state(getattr(scheduler, k), inner)


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def fit(adapter: ModelAdapter, log: Callable[[str], None] = print, resume_file: Path | None = None) -> dict:
    """Train with the adapter's FinBench settings and load the weights its selection rule keeps.

    After every epoch the validation split is evaluated (metrics in the history; the rules that
    need it use it). Returns {'history': DataFrame, 'kept_epoch', 'epochs_run', 'seconds'}; the
    history has one row per epoch: train_loss, lr, valid_* metrics, and kept (this epoch's weights
    became the kept ones).

    With `resume_file`, the whole training state (weights, optimizer, scheduler, random generators,
    history, kept weights) is saved there after every epoch, and if the file already exists training
    picks up after the last saved epoch. An interrupted run then loses at most one epoch, and ends
    with the same weights as an uninterrupted one. The caller deletes the file once the run is saved.
    """
    h, model = adapter.hparams, adapter.model
    rule = h["selection"]
    if rule not in SELECTION_RULES:
        raise ValueError(f"unknown selection rule {rule!r}")
    optimizer, scheduler = adapter.configure_optimizer()
    recent = collections.deque(maxlen=h.get("smooth_steps", 1))
    history, kept_state, kept_epoch = [], None, None
    best = np.inf if rule == "min_valid_loss" else -np.inf
    stall, start, elapsed, first_epoch, finished = 0, time.time(), 0.0, 0, False

    if resume_file is not None and Path(resume_file).exists():
        state = torch.load(resume_file, map_location="cpu", weights_only=False)
        device = next(model.parameters()).device
        on_device = lambda sd: {k: v.to(device) for k, v in sd.items()}  # noqa: E731
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            _load_scheduler_state(scheduler, state["scheduler"])
        recent.extend(on_device(sd) for sd in state["recent"])
        history, kept_epoch, best, stall = state["history"], state["kept_epoch"], state["best"], state["stall"]
        kept_state = on_device(state["kept_state"]) if state["kept_state"] is not None else None
        elapsed, finished, epoch = state["seconds"], state["finished"], state["epoch"]
        first_epoch = epoch + 1
        _load_rng_state(state["rng"])
        log(f"  resuming after epoch {epoch} ({elapsed / 60:.0f} min already trained)")

    def save_state(epoch: int, finished: bool) -> None:
        if resume_file is None:
            return
        state = {"epoch": epoch, "finished": finished, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": _scheduler_state(scheduler) if scheduler is not None else None,
                 "recent": list(recent), "history": history, "kept_state": kept_state, "kept_epoch": kept_epoch,
                 "best": best, "stall": stall, "seconds": elapsed + time.time() - start, "rng": _rng_state()}
        tmp = Path(resume_file).with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, resume_file)  # atomic: a kill during the save leaves the previous file intact

    for epoch in range(first_epoch, h["n_epochs"]) if not finished else ():
        tick = time.time()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row["train_loss"] = train_epoch(adapter, optimizer, scheduler, epoch)
        current = None
        if rule == "smoothed_valid_ic":
            current = copy.deepcopy(model.state_dict())
            recent.append(current)
            if epoch < h.get("eval_from_epoch", 0):  # DiscoverPLF evaluates only from epoch 20
                history.append(row | {"kept": False, "seconds": time.time() - tick})
                save_state(epoch, finished=False)
                continue
            model.load_state_dict(average_params(recent))

        _, _, valid = evaluate(adapter, "valid")
        row |= {f"valid_{k}": v for k, v in valid.items()}
        if rule == "min_valid_loss":
            row["valid_loss"] = validation_loss(adapter, "valid", epoch)

        score = {"min_valid_loss": row.get("valid_loss"), "smoothed_valid_ic": valid["IC"],
                 "pooled_valid_ic": valid["IC_pooled"]}.get(rule)
        improved = score is not None and (score < best if rule == "min_valid_loss" else score > best)
        if improved:
            best, kept_state, kept_epoch, stall = score, copy.deepcopy(model.state_dict()), epoch, 0
        elif score is not None:
            stall += 1
        if current is not None:
            model.load_state_dict(current)  # keep training the latest weights, not the average
        row |= {"kept": improved, "seconds": time.time() - tick}
        history.append(row)
        log(f"  epoch {epoch:3d}  train loss {row['train_loss']:.4f}  valid IC {valid['IC']:+.4f}  "
            f"RankIC {valid['RankIC']:+.4f}" + ("  *" if improved else "") + f"  ({row['seconds']:.0f}s)")

        stop = rule == "train_loss_threshold" and row["train_loss"] <= h["train_stop_loss_thred"]
        if not stop and h.get("early_stop") and stall >= h["early_stop"]:
            log(f"  early stop: {stall} epochs without improvement")
            stop = True
        save_state(epoch, finished=stop or epoch == h["n_epochs"] - 1)
        if stop:
            break

    if rule in ("last", "train_loss_threshold") or kept_state is None:  # None: the score was never finite
        if kept_state is None and rule not in ("last", "train_loss_threshold"):
            log("  no epoch improved the validation score: keeping the last weights")
        kept_state, kept_epoch = copy.deepcopy(model.state_dict()), epoch
        history[-1]["kept"] = True
    model.load_state_dict(kept_state)
    model.eval()
    return {"history": pd.DataFrame(history), "kept_epoch": kept_epoch, "epochs_run": epoch + 1,
            "seconds": elapsed + time.time() - start}
