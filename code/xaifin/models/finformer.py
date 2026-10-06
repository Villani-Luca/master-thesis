"""FinFormer: static-dynamic graph transformer (BigData 2023), copied from FinBench.

Source: code/finbench/Regression/FinFormer/model.py (modules unchanged), finformer.py (the xavier
initialization of Finformer_Model) and utils.py (edgeIndexTransform). FinBench is MIT licensed,
Copyright (c) 2026 softlab-unimore.

Two library functions FinBench imports are copied here instead of adding the libraries:
concordance_cc and pearsonr from audtorch 0.6.4 (audtorch/metrics/functional.py, MIT licensed),
the training loss; and to_undirected from torch_geometric 2.8 (utils/undirected.py), written
with torch.unique: it appends the reversed edges and removes duplicates, sorted by (row, col)
as torch_geometric's coalesce.

In eval mode the model is deterministic and differentiable w.r.t. x. Input: FinBench sorts the
360 Alpha360 columns alphabetically (Index.difference) and reshapes them to [N, 60, 6]
(issue 4 in docs/model_notes.md); the adapter takes x in the original column order and applies
the same permutation, so all Alpha360 adapters share one input space. The graph is held fixed.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from xaifin.data.datasets import DayBatch, finformer_splits
from xaifin.data.features import alpha360_names
from xaifin.models.base import ModelAdapter


def pearsonr(x, y, batch_first=True):
    """audtorch.metrics.functional.pearsonr (audtorch 0.6.4)."""
    assert x.shape == y.shape
    if batch_first:
        dim = -1
    else:
        dim = 0
    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)
    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)
    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    corr = bessel_corrected_covariance / (x_std * y_std)
    return corr


def concordance_cc(x, y, batch_first=True):
    """audtorch.metrics.functional.concordance_cc (audtorch 0.6.4)."""
    assert x.shape == y.shape
    if batch_first:
        dim = -1
    else:
        dim = 0
    bessel_correction_term = (x.shape[dim] - 1) / x.shape[dim]
    r = pearsonr(x, y, batch_first)
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    ccc = 2 * r * x_std * y_std / (x_std * x_std
                                   + y_std * y_std
                                   + (x_mean - y_mean)
                                   * (x_mean - y_mean)
                                   / bessel_correction_term)
    return ccc


def edge_index(adjacency: torch.Tensor) -> torch.Tensor:
    """FinBench's edgeIndexTransform: the nonzero entries of `adjacency` as an undirected edge list [2, E]."""
    row, col = adjacency.nonzero(as_tuple=True)
    edges = torch.stack([torch.cat([row, col]), torch.cat([col, row])])
    return torch.unique(edges, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class GatedFusion(nn.Module):
    def __init__(self, D: int):
        super(GatedFusion, self).__init__()
        self._fully_connected_xs = nn.Linear(D, D, bias=False)
        self._fully_connected_xt = nn.Linear(D, D)
        self._fully_connected_h = nn.Linear(D, D)
        self.leaky_relu = nn.LeakyReLU()

    def forward(
            self, HS: torch.FloatTensor, HT: torch.FloatTensor
    ) -> torch.FloatTensor:
        XS = self._fully_connected_xs(HS)
        XT = self._fully_connected_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self._fully_connected_h(H)
        H = self.leaky_relu(H)
        del XS, XT, z
        return H


class SpatialEmbeddingAggregation(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, dropout=dropout)

    def forward(self, x):
        # x -> [300, 60, 64] -> [60, 300, 64]
        src = x.transpose(1, 0)
        src = self.pos_encoder(src)
        output, _ = self.mha(src, src, src)
        out = output[-1]
        out = torch.layer_norm(out, normalized_shape=[self.d_model])

        return out


class SSA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=True)

    def masked_softmax(self, x, mask, eps=-1e10):
        x = x.masked_fill(~mask.bool(), eps)
        x = torch.softmax(x, dim=-1).squeeze()
        return x

    def forward(self, x, edge=None):
        DEVICE = x.device
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (x.shape[-1] ** -0.5)

        value = torch.FloatTensor(np.ones(edge.shape[1])).to(DEVICE)

        graph = torch.sparse.FloatTensor(indices=edge, values=value, size=attn[0].shape).to_dense().to(DEVICE)

        mask_rh = torch.where(graph >= 1, 1, 0)

        attn = torch.mul(attn, mask_rh).to(DEVICE)
        attn = self.masked_softmax(attn, mask_rh).squeeze()

        v = torch.matmul(attn, v)

        return v, attn


class DMHSA(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=True)
        self.num_heads = num_heads
        self.seq_len = 60

    def masked_softmax(self, x, mask, eps=-1e10):
        x = x.masked_fill(~mask.bool(), eps)
        x = torch.softmax(x, dim=-1).squeeze()
        return x

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(2, 3)) * (x.shape[-1] ** -0.5)
        mask_attn = attn

        dynamic_mask = torch.where(mask_attn >= torch.mean(mask_attn, dim=-1, keepdim=True), 1, 0).reshape(B,
                                                                                                           self.num_heads,
                                                                                                           N, N)
        attn = torch.mul(attn, dynamic_mask)
        attn = self.masked_softmax(attn, dynamic_mask).squeeze()
        v = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)

        return v, attn


class SparseSDTransformerLayer(nn.Module):
    def __init__(self, d_model=20, n_heads=2, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.layer_norm_eps = 1e-5
        self.hidden_size = 64
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.mlp_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.dmhsa = DMHSA(dim=self.hidden_size, num_heads=n_heads)
        self.ssa = SSA(dim=self.hidden_size)

    def forward(self, x, edge):
        out = x
        dynamic_emb, mask = self.dmhsa(out)
        static_emb, _ = self.ssa(out, edge)
        value = dynamic_emb.squeeze() + static_emb.squeeze()
        residual = value + x
        out = self.layer_norm1(residual)
        out = self.mlp_out(out)
        out = self.layer_norm2(out + residual)
        return out, mask


class SparseSDTransformerEncoder(nn.Module):
    def __init__(self, d_feat, hidden_size, num_layers, batch_first=True, n_head=2, seq_len=60):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.n_head = n_head
        self.seq_len = seq_len
        self.sparse_trans1 = SparseSDTransformerLayer(d_model=d_feat, n_heads=n_head)
        self.relu = nn.LeakyReLU()

    def forward(self, x, edge_index):
        # [300,60,64]
        out = x
        out = out.transpose(1, 0)
        out, mask = self.sparse_trans1(out, edge_index)
        out = out.transpose(1, 0)
        out = self.relu(out)
        return out, mask


class TemporalEncoder(nn.Module):
    def __init__(self, d_feat=6, d_model=64, num_layers=2, dropout=0.5):
        super(TemporalEncoder, self).__init__()
        self.dropout = dropout
        self.rnn = nn.GRU(d_feat, d_model, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.relu = nn.LeakyReLU()

    def forward(self, src):
        out, _ = self.rnn(src)
        out = self.relu(out)
        return out.squeeze()


class Finformer(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, seq_len=60, temporal_dropout=0.0, snum_head=2, num_layers=2):
        super(Finformer, self).__init__()
        self.fc_in = nn.Linear(d_feat, hidden_size)
        self.ssdt = SparseSDTransformerEncoder(d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers,
                                               batch_first=True, n_head=snum_head, seq_len=seq_len)
        self.te = TemporalEncoder(d_feat=hidden_size, d_model=hidden_size, num_layers=num_layers,
                                  dropout=temporal_dropout)
        self.gate_fusion = GatedFusion(hidden_size)
        self.s_aggr = SpatialEmbeddingAggregation(d_model=hidden_size, dropout=temporal_dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index):
        x = self.fc_in(x)
        ts = self.te(x)
        res_ts = ts[:, -1, :].squeeze()

        # Static-Dynamic
        # [300,60,64]
        move, mask = self.ssdt(x, edge_index)
        out = self.s_aggr(move)
        out = self.gate_fusion(out, res_ts)
        out = self.linear(out)
        return out.squeeze(), mask.squeeze()


# FinBench's column order: Index.difference sorts the 360 names alphabetically (CLOSE0, CLOSE1, CLOSE10, ...).
FINBENCH_ORDER = [alpha360_names().index(c) for c in sorted(alpha360_names())]


class FinFormerAdapter(ModelAdapter):
    """FinFormer as FinBench trains it (Regression/FinFormer/train.py, finformer.py).

    Data: finformer_splits (Alpha360 rows, adjacency extra); batch.y is already the daily z-score
    of the forward return over pred_len days. Training: loss = -CCC(prediction, label); Adam, no
    gradient clipping, up to 200 epochs. FinBench keeps the epoch with the best validation IC,
    computed over all stock-days pooled (not a daily mean), and stops after 10 epochs without
    improvement. Like MATCC it seeds only the CUDA generator (issue 8); the adapter seeds all.
    """

    name = "FinFormer"
    group = "alpha360"
    HPARAMS = {
        "d_feat": 6, "hidden_size": 64, "temporal_dropout": 0.4, "snum_head": 4,
        "lr": 2e-4, "n_epochs": 200, "grad_clip": None, "scheduler_step": None, "early_stop": 10,
    }

    @property
    def feature_names(self) -> list[str]:
        return alpha360_names()

    def build_model(self) -> nn.Module:
        h = self.hparams
        model = Finformer(d_feat=h["d_feat"], hidden_size=h["hidden_size"], temporal_dropout=h["temporal_dropout"],
                          snum_head=h["snum_head"]).to(self.device)
        # Finformer_Model.__init__ (finformer.py) moves the model to the device first, so on a GPU the
        # xavier weights come from the CUDA generator: the initial weights depend on the device.
        for n in model.modules():
            if isinstance(n, nn.Linear):
                n.weight = nn.init.xavier_normal_(n.weight, gain=1.)
        return model

    def load_splits(self) -> dict:
        c = self.cfg
        return finformer_splits(c.universe, c.test_year, c.pred_len, clean=c.clean)

    def _inputs(self, x: torch.Tensor, extras: dict) -> tuple:
        x = x.to(self.device)[:, -1, FINBENCH_ORDER]
        return x.reshape(len(x), -1, self.hparams["d_feat"]), edge_index(extras["adjacency"]).to(self.device)

    def model_forward(self, x: torch.Tensor, extras: dict) -> torch.Tensor:
        return self.model(*self._inputs(x, extras))[0]

    def training_loss(self, batch: DayBatch, epoch: int) -> torch.Tensor:
        pred, _ = self.model(*self._inputs(batch.x, batch.extras))
        return -concordance_cc(pred, batch.y.to(self.device))

    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"]), None
