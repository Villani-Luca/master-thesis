"""FactorVAE (Duan et al., AAAI 2022), copied from FinBench with a corrected prediction readout.

Source: code/finbench/Regression/FactorVAE/module.py
FinBench is MIT licensed, Copyright (c) 2026 softlab-unimore.

The modules and the training path (`FactorVAE.forward(x, returns)`) are unchanged, so training
behaves exactly as in FinBench and FinBench state_dicts load as is. Changes (docs/model_notes.md):

- Issue 1, label leakage: FinBench's test()/validate() score `rec`, the reconstruction built from
  posterior factors that the encoder infers from the *true* returns. `predict()` uses only the
  prior path (feature extractor -> factor predictor -> decoder), which never sees the labels.
- Issue 2, random predictions: FinBench's decoder returns a sample (mu + eps * sigma) even in
  eval mode. `predict()` returns the mean, so predictions and attributions are deterministic.

Not changed here (it requires retraining): FinBench builds the label with the lookback length
instead of the horizon (issue 3, FactorVAE/train.py:87).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, num_latent, hidden_size, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.normalize = nn.LayerNorm(num_latent)
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: (N, seq_len, num_latent) -> stock latents (N, hidden_size)
        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)
        stock_latent, _ = self.gru(out)
        return stock_latent[:, -1, :]


class FactorEncoder(nn.Module):
    """Posterior factors from the future returns. Used in training only."""

    def __init__(self, num_factors, num_portfolio, hidden_size):
        super(FactorEncoder, self).__init__()
        self.num_factors = num_factors
        self.linear = nn.Linear(hidden_size, num_portfolio)
        self.softmax = nn.Softmax(dim=0)

        self.linear_mu = nn.Linear(num_portfolio, num_factors)
        self.linear_sigma = nn.Linear(num_portfolio, num_factors)
        self.softplus = nn.Softplus()

    def mapping_layer(self, portfolio_return):
        mean = self.linear_mu(portfolio_return.squeeze(1))
        sigma = self.softplus(self.linear_sigma(portfolio_return.squeeze(1)))
        return mean, sigma

    def forward(self, stock_latent, returns):
        weights = self.linear(stock_latent)
        weights = self.softmax(weights)  # (N, num_portfolio)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        portfolio_return = torch.mm(weights.transpose(1, 0), returns)  # (num_portfolio, 1)
        return self.mapping_layer(portfolio_return)


class AlphaLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AlphaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent):
        stock_latent = self.linear1(stock_latent)
        stock_latent = self.leakyrelu(stock_latent)
        alpha_mu = self.mu_layer(stock_latent)
        alpha_sigma = self.sigma_layer(stock_latent)
        return alpha_mu, self.softplus(alpha_sigma)


class BetaLayer(nn.Module):
    """Factor exposures beta (N, K)."""

    def __init__(self, hidden_size, num_factors):
        super(BetaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, num_factors)

    def forward(self, stock_latent):
        return self.linear1(stock_latent)


class FactorDecoder(nn.Module):
    def __init__(self, alpha_layer, beta_layer):
        super(FactorDecoder, self).__init__()
        self.alpha_layer = alpha_layer
        self.beta_layer = beta_layer

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def distribution(self, stock_latent, factor_mu, factor_sigma):
        """Mean and std of the returns y = alpha + beta @ z, without sampling.

        Same formulas as forward(), but deterministic and without in-place ops, so it can be
        differentiated w.r.t. the input for attribution methods.
        """
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1)
        factor_sigma = torch.where(factor_sigma == 0, torch.full_like(factor_sigma, 1e-6), factor_sigma)

        mu = alpha_mu + torch.matmul(beta, factor_mu)
        sigma = torch.sqrt(alpha_sigma**2 + torch.matmul(beta**2, factor_sigma**2) + 1e-6)
        return mu, sigma

    def forward(self, stock_latent, factor_mu, factor_sigma):
        # Training path, unchanged from FinBench (samples; modifies factor_sigma in place).
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1)
        factor_sigma[factor_sigma == 0] = 1e-6

        mu = alpha_mu + torch.matmul(beta, factor_mu)
        sigma = torch.sqrt(alpha_sigma**2 + torch.matmul(beta**2, factor_sigma**2) + 1e-6)
        return self.reparameterize(mu, sigma)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, stock_latent):
        self.key = self.key_layer(stock_latent)
        self.value = self.value_layer(stock_latent)

        attention_weights = torch.matmul(self.query, self.key.transpose(1, 0))  # (N)
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.key.shape[1]) + 1e-6)
        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights)
        attention_weights = F.softmax(attention_weights, dim=0)  # (N)

        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            return torch.zeros_like(self.value[0])
        return torch.matmul(attention_weights, self.value)  # (H)


class FactorPredictor(nn.Module):
    """Prior factors from the stock latents only (no future information)."""

    def __init__(self, hidden_size, num_factor):
        super(FactorPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_factor = num_factor
        self.attention_layers = nn.ModuleList([AttentionLayer(self.hidden_size) for _ in range(num_factor)])

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent):
        h_multi = torch.cat([layer(stock_latent) for layer in self.attention_layers], dim=0)
        h_multi = h_multi.view(self.num_factor, -1)
        h_multi = self.leakyrelu(self.linear(h_multi))
        pred_mu = self.mu_layer(h_multi).view(-1)
        pred_sigma = self.softplus(self.sigma_layer(h_multi)).view(-1)
        return pred_mu, pred_sigma


class FactorVAE(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        kl_div = (torch.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div

    def forward(self, x, returns):
        """Training step, unchanged from FinBench. Uses the true returns: never use its output as a prediction."""
        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        reconstruction = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)

        reconstruction_loss = F.mse_loss(reconstruction, returns)

        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        return vae_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma

    def predict_distribution(self, x):
        """Leak-free predicted return distribution: mean and std, each of shape (N,)."""
        stock_latent = self.feature_extractor(x)
        prior_mu, prior_sigma = self.factor_predictor(stock_latent)
        mu, sigma = self.factor_decoder.distribution(stock_latent, prior_mu, prior_sigma)
        return mu.squeeze(-1), sigma.squeeze(-1)

    def predict(self, x):
        """Leak-free, deterministic prediction (N,): the mean of the predicted return distribution.

        Use this for metrics, portfolios and explanations. Call model.eval() first so the
        attention dropout is off.
        """
        return self.predict_distribution(x)[0]


def build_factorvae(num_latent=157, hidden_size=64, num_factor=96, num_portfolio=128):
    """Build FactorVAE as FinBench's FactorVAE/train.py does (defaults = its argparse defaults)."""
    feature_extractor = FeatureExtractor(num_latent=num_latent, hidden_size=hidden_size)
    factor_encoder = FactorEncoder(num_factors=num_factor, num_portfolio=num_portfolio, hidden_size=hidden_size)
    factor_decoder = FactorDecoder(AlphaLayer(hidden_size), BetaLayer(hidden_size, num_factor))
    factor_predictor = FactorPredictor(hidden_size, num_factor)
    return FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)
