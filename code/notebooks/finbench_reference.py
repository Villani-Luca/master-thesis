"""FinBench's own code, run next to the xaifin adapters for the Step 2 equivalence test (notebook 02).

Test harness only: it loads code/finbench/ read-only, by file path (the package never imports
FinBench). For each model it builds FinBench's datasets with FinBench's data code, FinBench's
model, and calls FinBench's own training-step and evaluation functions. Two things are adapted,
because the scripts cannot run here as they are:

- the scripts are not imported (their module-level code reads ../../Evaluation/data and imports
  qlib, wandb, tensorboard, torch_geometric, audtorch, which are not installed); the functions the
  test needs are taken from their source with `function_source` / `class_source` and executed
  with the names they use;
- FinFormer's edgeIndexTransform uses torch_geometric.utils.to_undirected: it is replaced by
  `edge_index_transform` below (same edges; checked in the Step 2 adapter check).

The driver (`SeededDays`) fixes the order of the training days and reseeds torch before each day,
on both sides, so that dropout and sampling noise are the same. FinBench draws them from one
unseeded or seeded stream instead (issue 8 in docs/model_notes.md).
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from itertools import chain
from torch.utils.data import DataLoader

from xaifin.config import NATION

CODE = Path(__file__).resolve().parents[1]
FB = CODE / "finbench" / "Regression"
DATA = CODE / "data"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ loading FinBench code

def load_module(name: str, path: Path, search_path: Path | None = None):
    """Import a FinBench file under a private name; `search_path` resolves its own imports."""
    for clash in ("utils", "model", "dataloader", "load_dataset", "base_model", "module", "dataset"):
        sys.modules.pop(clash, None)
    if search_path:
        sys.path.insert(0, str(search_path))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        quiet(module)
    finally:
        if search_path:
            sys.path.remove(str(search_path))
    return module


def _source(path: Path, start: str) -> str:
    src = path.read_text(encoding="utf-8")
    i = src.index(start)
    ends = [j for j in (src.find("\ndef ", i + 1), src.find("\nclass ", i + 1), src.find("\nif __name__", i + 1)) if j > 0]
    return src[i:min(ends) if ends else len(src)]


def function_source(path: Path, name: str) -> str:
    return _source(path, f"def {name}(")


def class_source(path: Path, name: str) -> str:
    return _source(path, f"class {name}")


def run_sources(namespace: dict, path: Path, functions=(), classes=()) -> dict:
    """Execute the given top-level functions and classes of a FinBench file in `namespace`."""
    for name in functions:
        exec(function_source(path, name), namespace)
    for name in classes:
        exec(class_source(path, name), namespace)
    return namespace


class _SilentBar:
    """Stands in for tqdm in FinBench modules: iterates, accepts the progress-bar calls, prints nothing."""

    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass


def quiet(module):
    """Silence a FinBench module's progress bars."""
    if hasattr(module, "tqdm"):
        module.tqdm = _SilentBar
    return module


# ------------------------------------------------------------------ driving both sides alike

class SeededDays:
    """Items of a dataset in a fixed order; torch is reseeded with base_seed + k before the k-th."""

    def __init__(self, get_item, order, base_seed: int):
        self.get_item, self.order, self.base_seed = get_item, list(order), base_seed

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        for k, i in enumerate(self.order):
            torch.manual_seed(self.base_seed + k)
            yield self.get_item(i)


def state_diff(a: dict, b: dict) -> float:
    """Largest absolute difference between two state_dicts with the same keys."""
    assert list(a) == list(b)
    return max(float((a[k].float().cpu() - b[k].float().cpu()).abs().max()) for k in a if a[k].numel())


def filter_constituents(universe: str, test_start: str, source: str = "eodhd") -> list[str]:
    """FinBench's filter_constituents_by_date on the constituents file it reads."""
    utils = load_module("fb_master_utils", FB / "MASTER" / "utils.py")
    path = (DATA / "constituents" / "eodhd" / f"{universe}.csv" if source == "eodhd"
            else DATA / universe / f"{universe}_constituents.csv")
    return utils.filter_constituents_by_date(pd.read_csv(path), test_start)["EODHD"].tolist()


def _alpha158_frame(universe, w, label_len, market):
    """The table of MASTER/MATCC train.py (market=True) before normalization."""
    df = pd.read_csv(DATA / universe / f"{universe}_alpha158.csv")
    df = df[df["instrument"].isin(filter_constituents(universe, w.start_test_date))]
    if market:
        df = pd.merge(df, pd.read_csv(DATA / universe / f"{NATION[universe]}_market.csv"), how="left", on="date")
    close = pd.read_csv(DATA / universe / f"{universe}.csv")[["date", "instrument", "adj_close"]]
    close = close.sort_values(["instrument", "date"])
    close["Label"] = close.groupby("instrument")["adj_close"].transform(lambda x: (x.shift(-label_len) - x) / x)
    df = df.merge(close[["date", "instrument", "Label"]], on=["date", "instrument"], how="left")
    train = df[(df["date"] >= w.start_date) & (df["date"] <= w.end_train_date)]
    return df[df["instrument"].isin(train["instrument"].drop_duplicates().tolist())]


# ------------------------------------------------------------------ MASTER

class MASTERReference:
    """FinBench Regression/MASTER: CSVDataset, MASTERModel (SequenceModel.train_epoch, predict)."""

    def __init__(self, universe, w, seq_len, pred_len):
        ld = load_module("fb_master_ld", FB / "MASTER" / "load_dataset.py")
        self.master = load_module("fb_master", FB / "MASTER" / "master.py", FB / "MASTER")
        df = _alpha158_frame(universe, w, pred_len, market=True)
        z = ld.RobustZScoreNormalization(df[(df["date"] >= w.start_date) & (df["date"] <= w.end_train_date)])
        self.train = ld.CSVDataset(df, seq_len, pred_len, w.start_date, w.end_train_date, z)
        self.test = ld.CSVDataset(df, seq_len, pred_len, w.start_test_date, w.end_date, z, period="test")
        self.gate_end = df.shape[1] - 3  # = shape_m + shape_a - 3 of train.py

    def build(self):
        return self.master.MASTERModel(
            d_feat=157, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5, beta=5,
            n_epochs=40, lr=1e-5, gate_input_end_index=self.gate_end, gate_input_start_index=157,
            save_path=".", GPU=0, train_stop_loss_thred=0.95)

    def init_state(self):
        """state_dict of a freshly built model (call set_seed first)."""
        return self.build().model.state_dict()

    def load(self, state):
        self.model = self.build()
        self.model.model.load_state_dict(state)
        self.model.fitted = 1

    def train_epoch(self, order, base_seed):
        return self.model.train_epoch(SeededDays(lambda i: self.train[i].unsqueeze(0), order, base_seed))

    def predict_test(self):
        preds, labels, _ = self.model.predict(self.test, num_workers=0)
        return [p.ravel() for p in preds], [np.asarray(y).ravel() for y in labels]

    def state(self):
        return self.model.model.state_dict()


# ------------------------------------------------------------------ MATCC

class MATCCReference(MASTERReference):
    """FinBench Regression/MATCC: CSVDataset, MATCC, ChainedScheduler, train.py train_epoch / valid_epoch."""

    def __init__(self, universe, w, seq_len, pred_len):
        ld = load_module("fb_matcc_ld", FB / "MATCC" / "load_dataset.py")
        self.matcc = load_module("fb_matcc", FB / "MATCC" / "model" / "MATCC.py", FB / "MATCC")
        self.sched = load_module("fb_matcc_sched", FB / "MATCC" / "lr_scheduler.py")
        df = _alpha158_frame(universe, w, pred_len, market=True)
        z = ld.RobustZScoreNormalization(df[(df["date"] >= w.start_date) & (df["date"] <= w.end_train_date)])
        self.train = ld.CSVDataset(df, seq_len, pred_len, w.start_date, w.end_train_date, z)
        self.test = ld.CSVDataset(df, seq_len, pred_len, w.start_test_date, w.end_date, z, period="test")
        self.gate_end, self.seq_len = df.shape[1] - 3, seq_len
        from sklearn.metrics import r2_score
        self.ns = run_sources(dict(np=np, pd=pd, torch=torch, r2_score=r2_score, DataLoader=DataLoader,
                                   DailyBatchSamplerRandom=ld.DailyBatchSamplerRandom),
                              FB / "MATCC" / "train.py",
                              functions=("calc_ic", "_init_data_loader", "loss_fn", "zscore", "drop_extreme",
                                         "train_epoch", "valid_epoch"))

    def build(self):
        return self.matcc.MATCC(d_model=256, d_feat=157, seq_len=self.seq_len, t_nhead=4, S_dropout_rate=0.5,
                                gate_input_start_index=157, gate_input_end_index=self.gate_end)

    def init_state(self):
        """state_dict of a freshly built model (call set_seed first)."""
        return self.build().state_dict()

    def load(self, state):
        self.model = self.build().to(DEVICE)
        self.model.load_state_dict(state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-3)
        self.scheduler = self.sched.ChainedScheduler(self.optimizer, T_0=15, T_mul=1, eta_min=2e-5, last_epoch=-1,
                                                     max_lr=3e-4, warmup_steps=10, gamma=1.0, coef=1.0, step_size=3,
                                                     cosine_period=4)

    def train_epoch(self, order, base_seed):
        days = SeededDays(lambda i: self.train[i].unsqueeze(0), order, base_seed)
        return self.ns["train_epoch"](days, self.optimizer, self.scheduler, self.model, DEVICE)

    def predict_test(self):
        loader = self.ns["_init_data_loader"](self.test, shuffle=False, drop_last=False, num_workers=0)
        _, _, preds, labels = self.ns["valid_epoch"](loader, self.model, DEVICE)
        return [p.ravel() for p in preds], [y.ravel() for y in labels]

    def state(self):
        return self.model.state_dict()


# ------------------------------------------------------------------ FactorVAE

class FactorVAEReference:
    """FinBench Regression/FactorVAE: TSDataSampler loaders, module.py, model.py train / test."""

    def __init__(self, universe, w, seq_len, pred_len):
        self.ds = load_module("fb_fvae_dataset", FB / "FactorVAE" / "dataset.py")
        self.module = load_module("fb_fvae_module", FB / "FactorVAE" / "module.py")
        self.loops = load_module("fb_fvae_model", FB / "FactorVAE" / "model.py")
        df = pd.read_csv(DATA / universe / f"{universe}_alpha158.csv")
        df = df[df["instrument"].isin(filter_constituents(universe, w.start_test_date))].copy()
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        close = pd.read_csv(DATA / universe / f"{universe}.csv")[["date", "instrument", "adj_close"]]
        close = close.rename(columns={"date": "datetime"})
        close["datetime"] = pd.to_datetime(close["datetime"])
        close = close.sort_values(["instrument", "datetime"])
        close["Label"] = close.groupby("instrument")["adj_close"].transform(lambda x: (x.shift(-seq_len) - x) / x)
        df = df.merge(close[["datetime", "instrument", "Label"]], on=["datetime", "instrument"], how="left")
        train = df[(df["datetime"] >= w.start_date) & (df["datetime"] <= w.end_train_date)]
        df = df[df["instrument"].isin(train["instrument"].drop_duplicates().tolist())]
        df = df.set_index(["datetime", "instrument"]).reorder_levels(["datetime", "instrument"]).sort_index()
        dates = df.index.get_level_values("datetime")
        z = self.ds.RobustZScoreNormalization(df[(dates >= w.start_date) & (dates <= w.end_train_date)])
        df = z.robust_zscore(df)
        df = df[df["Label"].notna()].fillna(0)
        df["Label"] = df.groupby(level="datetime")["Label"].transform(
            lambda g: ((g - g.mean()) / g.std()).clip(lower=-3.0, upper=3.0))
        self.train = self.ds.init_data_loader(df, step_len=seq_len, shuffle=False, start=w.start_date, end=w.end_train_date)
        self.test = self.ds.init_data_loader(df, step_len=seq_len, shuffle=False, start=w.start_test_date, end=w.end_date)
        self.groups = list(self.train.batch_sampler.grouped_indices)

    def build(self):
        m = self.module
        fe = m.FeatureExtractor(num_latent=157, hidden_size=64)
        enc = m.FactorEncoder(num_factors=96, num_portfolio=128, hidden_size=64)
        alpha, beta = m.AlphaLayer(64), m.BetaLayer(64, 96)
        dec = m.FactorDecoder(alpha, beta)
        pred = m.FactorPredictor(64, 96)
        return m.FactorVAE(fe, enc, dec, pred)

    def init_state(self):
        """state_dict of a freshly built model (call set_seed first)."""
        return self.build().state_dict()

    def load(self, state):
        self.model = self.build().to(DEVICE)
        self.model.load_state_dict(state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train) * 30)

    def _day(self, i):
        dataset = self.train.dataset
        return self.ds.custom_collate_fn([dataset[j] for j in self.groups[i]])

    def train_epoch(self, order, base_seed):
        return self.loops.train(self.model, SeededDays(self._day, order, base_seed), self.optimizer, self.scheduler, None)

    def predict_test(self):
        """FinBench's test(): the posterior reconstruction `rec`, which reads the true labels (issue 1)."""
        _, _, preds, labels = self.loops.test(self.model, self.test, None)
        return [p.ravel() for p in preds], [y.ravel() for y in labels]

    def predict_test_mean(self):
        """FinBench's prediction(x) without its sampling noise (issue 2): the leak-free mean."""
        self.model.eval()
        self.model.factor_decoder.reparameterize = lambda mu, sigma: mu
        preds = []
        with torch.no_grad():
            for data, _ in self.test:
                preds.append(self.model.prediction(data[:, :, :-1].to(DEVICE).float()).cpu().numpy().ravel())
        del self.model.factor_decoder.reparameterize
        return preds

    def state(self):
        return self.model.state_dict()


# ------------------------------------------------------------------ HIST and DiscoverPLF

class HISTReference:
    """FinBench Regression/HIST: create_loaders, model.HIST, train.py train_epoch / test_epoch."""

    folder, constituents = "HIST", "universe"

    def __init__(self, universe, w, seq_len, pred_len):
        folder = FB / self.folder
        utils = load_module(f"fb_{self.folder}_utils", folder / "utils.py")
        dl = load_module(f"fb_{self.folder}_dl", folder / "dataloader.py")
        self.models = self._models(folder)
        args = SimpleNamespace(data_path=str(DATA), universe=universe, start_date=w.start_date,
                               end_train_date=w.end_train_date, start_valid_date=w.start_valid_date,
                               end_valid_date=w.end_valid_date, start_test_date=w.start_test_date, end_date=w.end_date,
                               pred_len=pred_len, batch_size=-1, pin_memory=True, model_name=self.folder,
                               beta_strategy="increasing", n_epochs=50)
        self.args = args
        ns = dict(pd=pd, np=np, torch=torch, device=DEVICE, DataLoader=dl.DataLoader,
                  RobustZScoreNormalization=dl.RobustZScoreNormalization, mse=utils.mse, metric_fn=utils.metric_fn,
                  filter_constituents_by_date=utils.filter_constituents_by_date,
                  select_valid_ticker=utils.select_valid_ticker, tqdm=_SilentBar, global_step=-1)
        if hasattr(utils, "cal_beta"):
            ns["cal_beta"] = utils.cal_beta
        self.ns = run_sources(ns, folder / "train.py",
                              functions=("loss_fn", "zscore", "train_epoch", "test_epoch", "create_loaders"))
        train, _, test, inc, *_ = self.ns["create_loaders"](args)
        self.train_loader, self.test_loader = train, test
        self.s2c = torch.Tensor(inc).to(DEVICE)

    def _models(self, folder):
        return load_module("fb_hist_model", folder / "model.py", folder)

    def build(self):
        return self.models.HIST(d_feat=6, num_layers=2, K=1)

    def init_state(self):
        """state_dict of a freshly built model (call set_seed first)."""
        return self.build().state_dict()

    def load(self, state):
        self.model = self.build().to(DEVICE)
        self.model.load_state_dict(state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)

    def _ordered_loader(self, order, base_seed):
        loader = self.train_loader
        days = SeededDays(lambda i: (i, slice(loader.daily_index[i], loader.daily_index[i] + loader.daily_count[i])),
                          order, base_seed)
        return SimpleNamespace(iter_batch=lambda: iter(days), batch_length=len(days), get=loader.get)

    def train_epoch(self, order, base_seed):
        self.ns["train_epoch"](self.model, self.optimizer, self._ordered_loader(order, base_seed), self.args, self.s2c)

    def predict_test(self):
        preds, labels, *_ = self.ns["test_epoch"](self.model, self.test_loader, self.args, self.s2c)
        return [p.ravel() for p in preds], [y.ravel() for y in labels]

    def state(self):
        return self.model.state_dict()


class DiscoverPLFReference(HISTReference):
    """FinBench Regression/DiscoverPLF: create_loaders, factormodel.FACTORMODEL, train.py train_epoch / test_epoch."""

    folder, constituents = "DiscoverPLF", "eodhd"

    def _models(self, folder):
        return load_module("fb_dplf_model", folder / "factormodel.py", folder)

    def build(self):
        return self.models.FACTORMODEL(d_feat=6, hidden_size=128, num_layers=1, dilations=[1, 2, 5], num_days=60, K=1)

    def train_epoch(self, order, base_seed, epoch=0):
        self.ns["train_epoch"](epoch, self.model, self.optimizer, self._ordered_loader(order, base_seed), self.args,
                               self.s2c)


# ------------------------------------------------------------------ FinFormer

def edge_index_transform(adj):
    """FinBench's edgeIndexTransform (FinFormer/utils.py) without torch_geometric: coo, then to_undirected."""
    rows, cols = np.nonzero(adj)
    edges = torch.tensor(np.vstack([np.r_[rows, cols], np.r_[cols, rows]]), dtype=torch.long)
    return torch.unique(edges, dim=1)


class FinFormerReference:
    """FinBench Regression/FinFormer: CustomDataset, model.Finformer, finformer.Finformer_Model.trainer / tester."""

    def __init__(self, universe, w, seq_len, pred_len):
        import copy, math
        from sklearn.metrics import r2_score
        from xaifin.models.finformer import concordance_cc  # = audtorch's (checked in the adapter check)
        folder = FB / "FinFormer"
        ld = load_module("fb_ff_ld", folder / "load_dataset.py")
        self.models = load_module("fb_ff_model", folder / "model.py")
        df = pd.read_csv(DATA / universe / f"{universe}_alpha360.csv")
        df = df[df["instrument"].isin(filter_constituents(universe, w.start_test_date))]
        close = pd.read_csv(DATA / universe / f"{universe}.csv")[["date", "instrument", "adj_close"]]
        close = close.sort_values(["instrument", "date"])
        close["Label"] = close.groupby("instrument")["adj_close"].transform(lambda x: (x.shift(-pred_len) - x) / x)
        df = df.merge(close[["date", "instrument", "Label"]], on=["date", "instrument"], how="left")
        train = df[(df["date"] >= w.start_date) & (df["date"] <= w.end_train_date)]
        df = df[df["instrument"].isin(train["instrument"].drop_duplicates().tolist())]
        z = ld.RobustZScoreNormalization(df[(df["date"] >= w.start_date) & (df["date"] <= w.end_train_date)])
        self.train = ld.CustomDataset(df, d_feat=6, pred_len=pred_len, start_date=w.start_date,
                                      end_date=w.end_train_date, z_score=z, period="train")
        self.test = ld.CustomDataset(df, d_feat=6, pred_len=pred_len, start_date=w.start_test_date,
                                     end_date=w.end_date, z_score=z, period="test")
        graph = np.load(DATA / universe / f"{universe}_sector_industry_matrix.npz")
        self.adj = (graph["adj_matrix"][:11].max(axis=0) > 0).astype(np.float32)
        self.tickers = graph["tickers"]
        ns = dict(np=np, pd=pd, torch=torch, nn=torch.nn, math=math, copy=copy, r2_score=r2_score, chain=chain,
                  tqdm=_SilentBar, pprint=print, Finformer=self.models.Finformer,
                  concordance_cc=concordance_cc, edgeIndexTransform=edge_index_transform)
        run_sources(ns, folder / "finformer.py", functions=("metric_fn_2",), classes=("Finformer_Model",))
        run_sources(ns, folder / "train.py", functions=("custom_collate",))
        self.ns = ns

    def build(self):
        return self.ns["Finformer_Model"](d_feat=6, hidden_size=64, num_layers=2, temporal_dropout=0.4, snum_head=4,
                                          device=DEVICE).finformer

    def init_state(self):
        """state_dict of a freshly built model (call set_seed first)."""
        return self.build().state_dict()

    def load(self, state):
        self.wrapper = self.ns["Finformer_Model"](d_feat=6, hidden_size=64, num_layers=2, temporal_dropout=0.4,
                                                  snum_head=4, device=DEVICE)
        self.wrapper.finformer.load_state_dict(state)
        self.optimizer = torch.optim.Adam(self.wrapper.finformer.parameters(), lr=2e-4)

    def train_epoch(self, order, base_seed):
        collate = self.ns["custom_collate"]
        self.wrapper.trainer(SeededDays(lambda i: collate([self.train[i]]), order, base_seed), self.optimizer,
                             self.adj, self.tickers)

    def predict_test(self):
        loader = DataLoader(self.test, shuffle=False, drop_last=False, collate_fn=self.ns["custom_collate"])
        _, _, _, preds, labels, _ = self.wrapper.tester(loader, self.adj, self.tickers)
        return [p.ravel() for p in preds], [y.ravel() for y in labels]

    def state(self):
        return self.wrapper.finformer.state_dict()
