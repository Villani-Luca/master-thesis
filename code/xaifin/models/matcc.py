"""MATCC: Market Trends and Cross-time Correlations (Cao et al., CIKM 2024), copied from FinBench.

Sources: code/finbench/Regression/MATCC/model/MATCC.py, RWKV.py (RWKV-LM, BlinkDL) and DLinear.py,
merged into one module with only the classes MATCC uses: Block keeps only its 'RWKV' branch, the one
MATCC builds. FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.

The modules are otherwise unchanged, so FinBench state_dicts load as they are. In eval mode the model
is deterministic and differentiable. Unlike MASTER, it reads the market features on every day of the
window (Filter), and adds them to the stock features instead of rescaling them.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from xaifin.data.features import market_names
from xaifin.models.master import N_ALPHA, MASTERAdapter
from xaifin.training.lr_scheduler import ChainedScheduler


def RWKV_Init(module, vocab_size, n_embd, rwkv_emb_scale):  # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters():  # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Linear):

                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == vocab_size and shape[1] == n_embd:  # final projection?
                    scale = rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == vocab_size and shape[1] == n_embd:  # token emb?
                    scale = rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale, 2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)  # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)


class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_attn, n_head, ctx_len):
        super().__init__()
        assert n_attn % n_head == 0
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_head = n_head
        self.head_size = n_attn // n_head

        with torch.no_grad():  # initial time_w curves for better convergence
            ww = torch.ones(n_head, ctx_len)
            curve = torch.tensor([-(ctx_len - 1 - i) for i in range(ctx_len)])  # the distance
            for h in range(n_head):
                if h < n_head - 1:
                    decay_speed = math.pow(ctx_len, -(h + 1) / (n_head - 1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)
        self.receptance = nn.Linear(n_embd, n_attn)

        # if .rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn()

        self.output = nn.Linear(n_attn, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]  # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60)  # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]


class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 5 * n_ffn // 2  # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)
        self.receptance = nn.Linear(n_embd, n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)  # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv


class Block(nn.Module):
    def __init__(self, n_embd, n_attn, n_head, ctx_len, n_ffn, hidden_sz, model_type="RWKV", layer_id=1):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if model_type == 'RWKV':
            self.attn = RWKV_TimeMix(layer_id, n_embd, n_attn, n_head, ctx_len)
            self.mlp = RWKV_ChannelMix(layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


def DLinear_Init(module, min_val=-5e-2, max_val=8e-2):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, min_val, max_val)
            if m.bias is not None:
                # 如果模型有偏置项，你也可以选择初始化它们
                nn.init.zeros_(m.bias)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size=8, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len, pred_len, enc_in, kernel_size, individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = enc_in

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


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

            atten_ave_matrixh = torch.softmax(torch.matmul(
                qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(
                atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Filter(nn.Module):
    def __init__(self, d_input, d_output, seq_len, kernel=5, stride=5):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.seq_len = seq_len

        self.trans = nn.Linear(d_input, d_output)

        self.aggregate = nn.Conv1d(
            d_output, d_output, kernel_size=kernel, stride=stride, groups=d_output)

        # 输入是[N, T, d_feat]
        conv_feat = math.floor((self.seq_len - kernel) / stride + 1)

        self.proj_out = nn.Linear(conv_feat, 1)

    def forward(self, x):
        x = self.trans.forward(x)  # [N, T, d_feat]
        x_trans = x.transpose(-1, -2)  # [N, d_feat, T]
        x_agg = self.aggregate.forward(x_trans)  # [N, d_feat, conv_feat]
        out = self.proj_out.forward(x_agg)  # [N, d_feat, 1]
        return out.transpose(-1, -2)  # [N, 1, d_feat]


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        # [N, 1, T], [N, T, D] --> [N, 1, D]
        output = torch.matmul(lam, z).squeeze(1)
        return output


class MATCC(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
                 seq_len=8, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.n_attn = d_model
        self.n_head = t_nhead

        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index -
                             gate_input_start_index)  # F'
        self.feature_gate = Filter(self.d_gate_input, self.d_feat, seq_len)

        self.rwkv = Block(layer_id=0, n_embd=self.d_model,
                          n_attn=self.n_attn, n_head=self.n_head, ctx_len=300,
                          n_ffn=self.d_model, hidden_sz=self.d_model)
        RWKV_Init(self.rwkv, vocab_size=self.d_model,
                  n_embd=self.d_model, rwkv_emb_scale=1.0)

        self.dlinear = DLinear(seq_len=seq_len, pred_len=seq_len,
                               enc_in=self.d_model, kernel_size=3, individual=False)
        DLinear_Init(self.dlinear, min_val=-5e-2, max_val=8e-2)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            self.dlinear,  # 【N,T,D】
            self.rwkv,  # 【N,T,D】

            SAttention(d_model=d_model, nhead=s_nhead,
                       dropout=S_dropout_rate),  # [T,N,D]

            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x[N, T, [股票自身的, 市场]]
        src = x[:, :, :self.gate_input_start_index]  # N, T, D
        gate_input = x[:, :, self.gate_input_start_index:self.gate_input_end_index]
        src = src + self.feature_gate.forward(gate_input)

        output = self.layers(src).squeeze(-1)

        return output


class MATCCAdapter(MASTERAdapter):
    """MATCC as FinBench trains it (Regression/MATCC/train.py).

    Same data, loss and scoring as MASTER (MATCC adopts MASTER's problem definition and FinBench
    copies its pipeline). Differences: Adam with weight decay, the ChainedScheduler stepped once per
    epoch (10 warm-up epochs, cosine restarts every 15, then step decay), 70 epochs, last epoch kept.
    FinBench seeds only the CUDA generator (train.py:424), so its initial weights on CPU are not
    reproducible; the adapter seeds every generator.
    """

    name = "MATCC"
    HPARAMS = {
        "d_model": 256, "n_head": 4, "dropout": 0.5,
        "lr": 3e-4, "weight_decay": 1e-3, "n_epochs": 70, "grad_clip": 3.0, "scheduler_step": "epoch",
        "warmup_epochs": 10, "T_0": 15, "T_mult": 1, "eta_min": 2e-5, "gamma": 1.0, "coef": 1.0,
        "cosine_period": 4, "step_size": 3,
    }

    def build_model(self) -> nn.Module:
        h = self.hparams
        return MATCC(d_model=h["d_model"], d_feat=N_ALPHA, seq_len=self.cfg.seq_len, t_nhead=h["n_head"],
                     S_dropout_rate=h["dropout"], gate_input_start_index=N_ALPHA,
                     gate_input_end_index=N_ALPHA + len(market_names(self.cfg.universe)))

    def configure_optimizer(self):
        h = self.hparams
        optimizer = torch.optim.Adam(self.model.parameters(), lr=h["lr"], betas=(0.9, 0.999),
                                     weight_decay=h["weight_decay"])
        scheduler = ChainedScheduler(optimizer, T_0=h["T_0"], T_mul=h["T_mult"], eta_min=h["eta_min"], last_epoch=-1,
                                     max_lr=h["lr"], warmup_steps=h["warmup_epochs"], gamma=h["gamma"],
                                     coef=h["coef"], step_size=h["step_size"], cosine_period=h["cosine_period"])
        return optimizer, scheduler
