"""MASTER: Market-Guided Stock Transformer (Li et al., AAAI 2024), copied from FinBench.

Source: code/finbench/Regression/MASTER/master.py (the nn.Modules; the MASTERModel training wrapper
is replaced by xaifin.training). FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.

The modules are unchanged, so FinBench and old master_model.ipynb state_dicts load as they are.
The model has no randomness in eval mode: `MASTER(x)` is already a deterministic, differentiable
[N, T, F] -> [N] map. Input layout: the 157 Alpha158 features, then the market gate features
(columns gate_input_start_index:gate_input_end_index, read on the last day of the window only).
"""

import math

import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from xaifin.data.datasets import DayBatch, alpha158_splits
from xaifin.data.features import alpha158_names, market_names
from xaifin.data.normalization import cs_zscore, drop_extreme
from xaifin.models.base import ModelAdapter

N_ALPHA = 157  # FinBench --gate_input_start_index


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index,
                 gate_input_end_index, beta):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]  # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        output = self.layers(src).squeeze(-1)

        return output


def build_master(n_market: int, d_model=256, t_nhead=4, s_nhead=2, dropout=0.5, beta=5) -> MASTER:
    """MASTER for inputs of N_ALPHA alpha features followed by `n_market` market features.

    The defaults are FinBench's Regression/MASTER/train.py defaults (training: lr 1e-5, 40 epochs).

    FinBench sets gate_input_end_index = (columns of the market file) + (columns of the alpha file) - 3,
    which is N_ALPHA + n_market.
    """
    return MASTER(d_feat=N_ALPHA, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=dropout,
                  S_dropout_rate=dropout, gate_input_start_index=N_ALPHA,
                  gate_input_end_index=N_ALPHA + n_market, beta=beta)


class MASTERAdapter(ModelAdapter):
    """MASTER as FinBench trains it (Regression/MASTER/train.py, base_model.py).

    Data: Alpha158 + market features, own-row windows (data.datasets.alpha158_splits); batch.y is
    the raw forward return over pred_len days. Training: drop the 2.5% label tails, z-score the
    rest, MSE; Adam, no scheduler, gradients clipped at 3. Scored against the daily z-score of the
    labels, without dropping tails. FinBench stops when the training loss falls below
    `train_stop_loss_thred` and keeps the last epoch; the Step 3 trainer decides the stopping rule.
    FinBench never passes --seed to MASTERModel (train.py:176), so its runs are not seeded at all;
    the adapter seeds every generator.
    """

    name = "MASTER"
    group = "alpha158"
    HPARAMS = {
        "d_model": 256, "t_nhead": 4, "s_nhead": 2, "dropout": 0.5, "beta": 5,
        "lr": 1e-5, "n_epochs": 40, "grad_clip": 3.0, "scheduler_step": None, "train_stop_loss_thred": 0.95,
    }

    @property
    def feature_names(self) -> list[str]:
        return alpha158_names() + market_names(self.cfg.universe)

    def build_model(self) -> nn.Module:
        h = self.hparams
        return build_master(len(market_names(self.cfg.universe)), h["d_model"], h["t_nhead"], h["s_nhead"],
                            h["dropout"], h["beta"])

    def load_splits(self) -> dict:
        c = self.cfg
        return alpha158_splits(c.universe, c.test_year, c.seq_len, c.pred_len, market=True, clean=c.clean)

    def model_forward(self, x: torch.Tensor, extras: dict) -> torch.Tensor:
        return self.model(x.to(self.device))

    def training_loss(self, batch: DayBatch, epoch: int) -> torch.Tensor:
        mask, label = drop_extreme(batch.y.to(self.device))
        label = cs_zscore(label)
        pred = self.model(batch.x.to(self.device)[mask])
        keep = ~torch.isnan(label)
        return torch.mean((pred[keep] - label[keep]) ** 2)

    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams["lr"]), None

    def target(self, batch: DayBatch) -> torch.Tensor:
        return cs_zscore(batch.y)
