"""The ModelAdapter interface: build, train, load and run any model the same way (THESIS_GUIDE.md §2.3).

Training, XAI, portfolios and the app only talk to adapters. An adapter owns one model for one run
(model, universe, rolling window, seed, T, L) and knows, for that model:

- how FinBench builds its data (`day_batches`, `target`),
- its prediction as a differentiable [N, T, F] -> [N] map (`forward`), which XAI explains,
- how FinBench trains it (`training_loss`, `configure_optimizer`, and the loop settings in
  `hparams`: n_epochs, grad_clip, scheduler_step), which the Step 3 trainer runs.
"""

import json
import random
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

from xaifin.config import CLEAN_DATA, CODE_ROOT, DEFAULT_SL_PL, RESULTS_ROOT
from xaifin.data.datasets import DayBatch


@dataclass
class RunConfig:
    """One training run: the cell of the grid (THESIS_GUIDE.md §2.2) plus hyper-parameter overrides."""

    model: str
    universe: str
    test_year: int
    seed: int = 42
    seq_len: int = DEFAULT_SL_PL[0]
    pred_len: int = DEFAULT_SL_PL[1]
    clean: bool = CLEAN_DATA
    hparams: dict = field(default_factory=dict)  # overrides of the adapter's HPARAMS

    @property
    def run_dir(self) -> Path:
        return (RESULTS_ROOT / "Regression" / self.model / self.universe / f"sl{self.seq_len}_pl{self.pred_len}"
                / f"seed{self.seed}" / f"y{self.test_year}")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA), as FinBench's FactorVAE/utils.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_hash() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class ModelAdapter(ABC):
    """Base class of the adapters. Subclasses set `name`, `group`, `HPARAMS` and the abstract methods.

    The constructor seeds every generator with `cfg.seed` and then builds the model, so the initial
    weights depend only on the seed. The data is loaded on first use of `day_batches`.
    """

    name: str
    group: str  # "alpha158" | "alpha360"
    HPARAMS: dict  # FinBench's defaults for this model; RunConfig.hparams overrides them

    def __init__(self, cfg: RunConfig, device: str | torch.device | None = None):
        self.cfg = cfg
        self.hparams = {**self.HPARAMS, **cfg.hparams}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        set_seed(cfg.seed)
        self.model = self.build_model().to(self.device)
        self._splits = None

    @property
    def seq_len(self) -> int:
        return self.cfg.seq_len

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """The F input features, in the order of the last axis of x."""

    @abstractmethod
    def build_model(self) -> nn.Module:
        """The untrained model, built from self.cfg and self.hparams."""

    @abstractmethod
    def load_splits(self) -> dict:
        """{'train', 'valid', 'test'} -> dataset of DayBatch, built as FinBench builds this model's data."""

    @abstractmethod
    def forward(self, x: torch.Tensor, extras: dict) -> torch.Tensor:
        """Predictions [N] for one day's inputs [N, T, F]: deterministic and differentiable.

        Call model.eval() first (see predict). `x` is moved to the adapter's device.
        """

    @abstractmethod
    def training_loss(self, batch: DayBatch) -> torch.Tensor:
        """FinBench's training loss on one day, label preprocessing included. Model in train mode."""

    @abstractmethod
    def configure_optimizer(self) -> tuple[torch.optim.Optimizer, object | None]:
        """FinBench's optimizer and learning-rate scheduler (None if it has none).

        hparams['scheduler_step'] says when the trainer steps the scheduler: 'epoch' or 'batch'.
        """

    def target(self, batch: DayBatch) -> torch.Tensor:
        """The labels FinBench scores the predictions against (test metrics). Default: batch.y."""
        return batch.y

    def splits(self) -> dict:
        if self._splits is None:
            self._splits = self.load_splits()
        return self._splits

    def day_batches(self, split: str, shuffle: bool = False) -> Iterator[DayBatch]:
        """The days of a split, in date order or shuffled with NumPy's generator (as FinBench)."""
        dataset = self.splits()[split]
        order = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(order)
        for i in order:
            yield dataset[i]

    @torch.no_grad()
    def predict(self, batch: DayBatch) -> np.ndarray:
        """Predictions [N] for one day, in eval mode, as a NumPy array."""
        self.model.eval()
        return self.forward(batch.x, batch.extras).cpu().numpy()

    def load(self, run_dir: Path | None = None) -> None:
        """Load model.pth from `run_dir` (default: cfg.run_dir)."""
        path = Path(run_dir or self.cfg.run_dir) / "model.pth"
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, run_dir: Path | None = None) -> None:
        """Write model.pth and config.json to `run_dir` (default: cfg.run_dir)."""
        run_dir = Path(run_dir or self.cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), run_dir / "model.pth")
        config = {**asdict(self.cfg), "hparams": self.hparams, "group": self.group,
                  "feature_names": self.feature_names, "git_hash": git_hash()}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
