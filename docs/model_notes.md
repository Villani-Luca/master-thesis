# Model notes: the six alpha-based regression models

Step 0.2 of [`THESIS_GUIDE.md`](../THESIS_GUIDE.md). One note per model: what it predicts, its inputs, normalizations, loss, whether it is cross-sectional, and which internal signals can be inspected for the XAI work. Each note compares the **paper** with the **FinBench implementation** (`code/finbench/Regression/<MODEL>/`), because FinBench is the code the thesis trains. Line references point into that folder. The thesis prose version is in `latex-thesis/chapters/02_second_chapter.tex`.

## Summary

| | MASTER | MATCC | FactorVAE | HIST | DiscoverPLF | FinFormer |
|---|---|---|---|---|---|---|
| Venue | AAAI 2024 | CIKM 2024 | AAAI 2022 | arXiv 2021 | TKDE 2024 | IEEE BigData 2023 |
| Features | Alpha158 + market | Alpha158 + market | Alpha158 | Alpha360 | Alpha360 | Alpha360 |
| Input per stock (FinBench) | `[T, 157+M]` | `[T, 157+M]` | `[T, 157]` | 360-vector (T=1) | 360-vector (T=1) | 360-vector (T=1) |
| Extra inputs | – | – | – | stock→concept matrix, market cap | stock→concept matrix, market cap | sector/industry adjacency |
| Backbone | Transformer (intra-stock) | RWKV (intra-stock) | GRU + VAE | 2-layer GRU | GRU/dilated + VAE | GRU (temporal) + sparse static-dynamic transformer (spatial) |
| Cross-sectional mixing | inter-stock attention | inter-stock attention + market trend | global attention in factor predictor | predefined + hidden concept graphs | concept graphs (HIST-based) | static + dynamic spatial attention |
| Loss (FinBench) | MSE | MSE | MSE recon. + KL | MSE | recon. + pred. + KL | −CCC |
| Built-in explanation | market gate α(m), T/S attention | T/S attention, trend vs fluctuation | ŷ = α + β·z | ŷ = predefined + hidden + individual | latent signal weights | fusion gate, spatial attention |

`M` = market gate features: **63 for US** universes, **42 for EU** (`<nation>_market.csv`). **Alpha158 has 157 features** in FinBench (no `VWAP0`; see `00_environment_check.ipynb`).

**Label (every model, FinBench):** forward return on adjusted close, `r = (close[t+L] − close[t]) / close[t]` over the horizon `L = pred_len`, normalized cross-sectionally per day. Exception: FactorVAE, see ⚠️ below. FinBench protocol (paper §4.1): (T, L) ∈ {(5,1), (20,5), (60,20)}; Alpha360 models use an effective lookback of T = 1 because the vector already covers 60 days; seeds {0, 5, 42}; rolling 4y train / 1y valid / 1y test with test years 2020–2024.

---

## ⚠️ Implementation issues found in FinBench

These were found by reading the code and, for the layout issue, by applying each model's reshape to the real column names of `dji_alpha360.csv`. They affect how comparable the six models are.

**Decision (2026-09-29): train exactly as FinBench does; fix only the prediction readout and file paths**, i.e. everything that doesn't change what the model learns. Fixes live in `xaifin` (FinBench stays read-only) and are covered by `code/tests/`.

| # | Issue | Status |
|---|---|---|
| 1 | FactorVAE test predictions use the labels | ✅ fixed: `FactorVAE.predict()` in `xaifin/models/factorvae.py` |
| 2 | FactorVAE predictions are random samples | ✅ fixed: `predict()` returns the mean |
| 3 | FactorVAE label uses `seq_len` | ⏸ kept as FinBench (changes training). FactorVAE's target is a T-day return; state it wherever FactorVAE is compared |
| 4 | Alpha360 column layout | ⏸ kept as FinBench (changes training). Attribute on the original 360 columns; possible *Seconda parte* experiment |
| 5 | MATCC market-file path | ✅ fixed: `xaifin.data.loading.market_path()` |
| 6 | FinFormer label ≠ README | ✅ nothing to fix (documentation only) |
| 7 | FactorVAE leaves KMID and KLEN unnormalized | ⏸ kept as FinBench (changes training). Found 2026-10-06 |
| 8 | MASTER, MATCC and FinFormer runs are not seeded | ⏸ the adapters seed every generator (no effect on what is learned). Found 2026-10-06 |
| 9 | HIST and DiscoverPLF run only on a GPU | ✅ fixed: `x.device` in `xaifin/models/hist.py`, `discoverplf.py`. Found 2026-10-06 |
| 10 | Training-loop differences across models | ⏸ kept: decided 2026-10-06 (Option A), every model trains and keeps its weights as in FinBench (`training/trainer.py`). Found 2026-10-06 |

Proof that the copied FactorVAE is exact: same `state_dict` keys as FinBench, identical training-forward outputs under the same seed (train and eval mode), and `predict()` equal to FinBench's `prediction()` without the sampling noise (one-off check, 2026-09-29).

1. **FactorVAE test predictions use the future returns (label leakage).** `FactorVAE/model.py:95` calls `factor_model(inputs, label)`, and line 100 stores `rec` as the prediction. `rec` is the **posterior** reconstruction, whose factors the encoder builds from the true returns (`module.py:246–255`). The leak-free path `FactorVAE.prediction(x)` (`module.py:269`) is never called. `validate()` does the same (`model.py:54, 59`). FinBench's FactorVAE test metrics are therefore likely optimistic.
2. **FactorVAE predictions are random.** `FactorDecoder.forward` returns `self.reparameterize(mu, sigma)` (`module.py:119`), a sample, even in eval mode. For XAI and portfolios, use the deterministic mean `μ = α_μ + β·μ_prior`.
3. **FactorVAE label horizon uses `seq_len`, not `pred_len`.** `FactorVAE/train.py:87`: `extract_labels(dataset, args, pred_len=args.seq_len)`. With (T=20, L=5) the target becomes a 20-day return while the other models predict 5-day returns.
4. **The Alpha360 column layout doesn't match the models' reshape.** FinBench's `*_alpha360.csv` is **lag-major**: `CLOSE0, OPEN0, HIGH0, LOW0, VOLUME0, VWAP0, CLOSE1, …` (`Evaluation/features/alpha360.py:40–53`). The models assume Qlib's **feature-major** layout:
   - HIST (`model.py:78–79`) and DiscoverPLF (`dataloader.py:92`, `factormodel.py:107`) reshape to `[6, 60]` and permute to `[60 steps, 6]`. GRU step 0 reads `CLOSE0, CLOSE10, …, CLOSE50`, step 1 reads `OPEN0, OPEN10, …`, step 6 reads `CLOSE1, CLOSE11, …`, and step 59 reads `VWAP9, …, VWAP59`. Each step holds one series at six lags ten days apart, cycling through the series, and the most recent day enters first.
   - FinFormer first **sorts the columns alphabetically** (`load_dataset.py:57`, `Index.difference`), then reshapes to `[60, 6]`. Model "timesteps" 0–9 = CLOSE lags in lexicographic order (`CLOSE0, CLOSE1, CLOSE10, …`), 10–19 = HIGH, and so on.

   The models still receive all 360 values in a consistent order, so they can learn, but not in chronological order: the temporal inductive bias of the GRU/attention layers doesn't apply as designed. For XAI, attribute on the **original 360 columns** and map to (series, lag) afterwards. That stays valid, but "temporal" readings of internal states or attention would not.
5. **MATCC reads the market file from two different paths**: `train.py:232` uses `{data_path}/{nation}_market.csv`, while `train.py:416` uses `{data_path}/{universe}/{nation}_market.csv`. The local data only has the second.
6. **FinFormer's label is a cross-sectional z-score**, not CSRank as the FinBench README states (`load_dataset.py:21, 64`).
7. **FactorVAE doesn't normalize its first two features.** `FactorVAE/train.py:102` moves `datetime` and `instrument` into the index before building `RobustZScoreNormalization`, which still takes `df.columns[2:-1]` (`dataset.py:32`), so the z-score skips KMID and KLEN instead of the two key columns. They enter the model raw (KMID ranges from −0.31 to 1.55 in dji's y2020 training data, while the normalized features are clipped to [−3, 3]). Checked on dji y2020.
8. **FinBench's MASTER and MATCC runs are not seeded.** MASTER's `train.py:176` never passes `--seed` to `MASTERModel`, so `SequenceModel` gets `seed=None` and seeds nothing; the seed only names the output folder. MATCC only calls `torch.cuda.manual_seed` (`train.py:424`), so its initial weights (built on the CPU) and the shuffling of the training days are not reproducible. FinBench's seed-to-seed spread for these two models is therefore uncontrolled randomness, and the Step 2 equivalence test can't expect identical training runs for them. FinFormer has the same problem (`FinFormer/train.py:68`, `torch.cuda.manual_seed` only).
9. **HIST and DiscoverPLF crash on the CPU.** Their forward passes call `torch.device(torch.get_device(x))` (`HIST/model.py:77`, `DiscoverPLF/factormodel.py:103, 354`), and `get_device` returns −1 for CPU tensors. The xaifin copies use `x.device`, which changes nothing on the GPU (checked: identical outputs).
10. **Each model is trained and selected differently**, beyond its architecture: MASTER stops when the training loss falls below 0.95 and keeps the last epoch; MATCC trains 70 epochs and keeps the last; FactorVAE keeps the lowest validation loss; HIST and DiscoverPLF keep the best validation IC of the weights averaged over the last 5 epochs (DiscoverPLF only from epoch 20); FinFormer keeps the best validation IC pooled over all stock-days, not averaged per day. Losses differ too (MSE, VAE losses, −CCC), and HIST ignores `--hidden_size` (it runs with 64, not 128). Data details also differ: HIST reads `<universe>_constituents.csv`, which maps nasdaq100's Ctrip to `CTRPX.US` instead of `TCOM.US`; HIST and DiscoverPLF drop stock-days without a market capitalization; FactorVAE and FinFormer don't cut the last training labels, which read validation-period prices.

---

## MASTER: Market-Guided Stock Transformer
*Li et al., AAAI 2024. FinBench: `Regression/MASTER/`.*

- **Predicts:** the daily cross-sectionally z-scored forward return ratio. The paper frames the target as encoding the stocks' *ranking*. It jointly predicts all stocks of one date (one batch = one day).
- **Inputs:** per stock, a `[T, F]` window of Alpha158 features, plus a **market status vector** `m` built from index price and volume (current value, mean and std over past windows) → FinBench `<nation>_market.csv` columns appended after the 157 alpha features (`gate_input_start_index=157`).
- **Architecture:** (1) **market-guided gate**, `α(m) = F · softmax((W m + b) / β)` with temperature β (`master.py:146–156`), which rescales every feature (`x̃ = α(m) ⊙ x`) and is shared by all stocks and timesteps of a date; (2) intra-stock Transformer across time; (3) inter-stock multi-head attention at each timestep; (4) temporal attention pooling with the last timestep as query; (5) linear predictor.
- **Normalization:** features → robust z-score (median/MAD) fit on the training window, clipped to [−3, 3]; label → drop the 2.5% tails, then daily cross-sectional z-score (training only, `base_model.py:96–98`).
- **Loss:** MSE.
- **Cross-sectional:** yes, through inter-stock attention, so stock *i*'s prediction depends on the other stocks' inputs that day.
- **Signals to inspect:** the **gate vector α(m)** is a built-in, market-conditioned global feature importance (one weight per Alpha158 feature per day), directly comparable with SHAP/IG rankings. Also intra-stock attention (which lags), inter-stock attention (which stocks), and temporal pooling weights. The attention modules compute their softmax but discard it (`master.py:73, 132`), so patch them to return it.

## MATCC: Market Trends and Cross-time Correlations
*Cao et al., CIKM 2024. FinBench: `Regression/MATCC/`.*

- **Predicts:** the same target and formulation as MASTER (it adopts MASTER's problem definition).
- **Inputs:** Alpha158 window + the same market features as MASTER (`gate_input_start_index=157`, end index derived from the market file).
- **Architecture:** (1) **market trend guidance**: a depthwise 1-D convolution over the market features gives a trend vector that is **added** to every stock's features (not a multiplicative gate as in MASTER); (2) **trend/fluctuation decomposition**: moving-average trend + residual fluctuation, each through its own linear map, then summed; (3) intra-stock **RWKV** (causal, linear-time attention-like RNN, no positional encoding); (4) inter-stock self-attention per timestep; (5) temporal attention pooling (as MASTER); (6) linear predictor.
- **Normalization:** robust z-score (median/MAD) on features; drop the 2.5% label tails; daily cross-sectional z-score of the label (`train.py:155–157`).
- **Loss:** MSE.
- **Cross-sectional:** yes (inter-stock attention).
- **Signals to inspect:** inter-stock and temporal attention; the relative magnitude of the **trend vs fluctuation branches** (does the model rely on smoothed trend or on noise?); the market-trend vector. There's no per-feature gate, so feature importance must come from post-hoc methods.

## FactorVAE: probabilistic dynamic factor model
*Duan et al., AAAI 2022. FinBench: `Regression/FactorVAE/`.*

- **Predicts:** cross-sectional stock returns as a **distribution**. The predicted return is Gaussian with mean `μ_pred(i) = α_μ(i) + Σ_k β(i,k) μ_prior(k)`, and the model also gives a std (a risk estimate).
- **Inputs:** Alpha158 window `[T, 157]` (`num_latent=157`), no market features.
- **Architecture:** GRU **feature extractor** → stock latents `e`. **Factor encoder** (training only): builds portfolios from `e`, uses the *future* returns to infer posterior factors `z_post`. **Factor decoder**: `ŷ = α(e) + β(e)·z` (alpha = idiosyncratic part, beta = factor exposures). **Factor predictor**: multi-head global attention over all stocks → prior factors `z_prior` (mean, std). At inference: extractor → predictor → decoder.
- **Normalization:** robust z-score on features; drop rows with NaN labels, fillna(0); daily cross-sectional z-score of the label clipped to ±3 (`train.py:95–100`).
- **Loss:** paper = negative log-likelihood of the reconstruction + γ·KL(posterior ‖ prior). FinBench = **MSE** reconstruction + KL (`module.py:246–265`).
- **Cross-sectional:** yes (global attention in the factor predictor; factors are shared by all stocks).
- **Signals to inspect:** the prediction is **exactly additive**, `ŷ_i = α_i + Σ_k β_ik z_k`, so each prediction splits into an idiosyncratic part and per-factor contributions. The model also exposes the predicted std (uncertainty, useful for Step 9) and the predictor's attention. See issues 1–3 above: the adapter must use `prediction(x)` with the mean, not FinBench's `rec`.

## HIST: concept-oriented shared information
*Xu et al., arXiv 2110.13716 (2021). FinBench: `Regression/HIST/`.*

- **Predicts:** the daily cross-sectionally z-scored forward return (Qlib-style stock trend).
- **Inputs:** one Alpha360 vector per stock (6 series × 60 lags; T = 1 in FinBench), plus a **stock→concept matrix** (`<universe>_inc_matrix.npz`) and **market capitalization** (`<universe>_market_cap.csv`, in $bn), which weights the stocks when building concept representations.
- **Architecture:** 2-layer GRU encoder → three modules in a doubly-residual chain: **predefined concepts** (graph over given sector/industry concepts), **hidden concepts** (mined from the residual, via stock–stock similarity), **individual** information (what remains). The paper sums the three forecasts. FinBench sums their hidden outputs and applies one linear layer (`model.py:149–150`); since that layer is linear, the prediction still **decomposes exactly** into three contributions.
- **Normalization:** robust z-score on features (fit on train), then fillna(0); label daily cross-sectional z-score (`train.py:112, 138`); NaN labels masked in the loss.
- **Loss:** MSE.
- **Cross-sectional:** yes, strongly: concepts aggregate information from other stocks weighted by market cap.
- **Signals to inspect:** the **predefined / hidden / individual contribution split** (how much of each prediction comes from shared vs stock-specific information; a direct handle on "data vs model"); concept attention weights; mined hidden concepts. Layout issue 4 applies.

## DiscoverPLF: predictable latent factors
*Hou et al., IEEE TKDE 36(10), 2024 (online 2023). FinBench: `Regression/DiscoverPLF/`.*

- **Predicts:** the same target as HIST. The paper's stock experiments plug its factor module into HIST ("Ours+HIST"), and FinBench's code keeps the HIST concept machinery.
- **Inputs:** Alpha360 vector (T = 1), stock→concept matrix, market cap (as HIST).
- **Architecture:** VAE-style inference of **latent factors** grouped into several independent, easily predictable **signal components**, built with different temporal strides/dilations (`--stride 1+2+5`). The components are forecast with simple dynamics, then combined. Disentanglement is encouraged with a total-correlation term with a discriminator (`discriminator.py`), then fed through HIST-like concept modules.
- **Normalization:** as HIST (robust z-score + daily cross-sectional z-scored label).
- **Loss:** reconstruction + prediction (MSE) + KL (+ total-correlation term); `--beta_strategy` (default `increasing`) schedules the KL weight.
- **Cross-sectional:** yes (concept graphs, as HIST).
- **Signals to inspect:** the mixing weights `alpha` over the signal components (`factormodel.py:72–84`), i.e. which temporal scale drives the prediction; the latent factors themselves; HIST-style concept contributions. Layout issue 4 applies.

## FinFormer: static-dynamic spatiotemporal transformer
*Zu et al., IEEE BigData 2023, pp. 1460–1469. FinBench: `Regression/FinFormer/`.*

- **Predicts:** the daily cross-sectionally z-scored forward return (FinBench, `load_dataset.py:64`).
- **Inputs:** Alpha360 vector reshaped to `[60, 6]` (see issue 4), plus a **static adjacency** from `<universe>_sector_industry_matrix.npz` (first 11 relation matrices collapsed to a binary graph, `train.py:22–23, 122–124`).
- **Architecture:** GRU temporal encoder per stock (`TemporalEncoder`, as in the paper; last hidden state = temporal embedding); **sparse static-dynamic transformer** across stocks (static = predefined sector/industry edges, dynamic = learned attention, sparsified); **gated spatio-temporal fusion** `z = σ(X_S + X_T)` mixing the spatial and temporal representations (`model.py:23–41`); linear head.
- **Normalization:** robust z-score on features (clipped ±3); label daily cross-sectional z-score.
- **Loss:** negative **concordance correlation coefficient** (`finformer.py:88`), which optimizes agreement in correlation *and* scale rather than squared error.
- **Cross-sectional:** yes (static and dynamic spatial attention).
- **Signals to inspect:** the **fusion gate `z`** (per stock and day, how much the prediction relies on other stocks vs the stock's own history, a natural data-vs-model probe); static vs dynamic spatial attention. Anything read along the 60-step axis (GRU states, the spatial transformer's sequence dimension) is only meaningful after fixing issue 4.

---

## What this means for the XAI framework (Steps 2 and 5)

- **All six models are cross-sectional:** explain a whole day at once and attribute stock *i*'s output to its own inputs while holding the others fixed (§2.3 of the guide). Keep the extra inputs (concept matrix, market cap, adjacency) fixed and documented.
- **Four models have built-in decompositions** that can serve as a check on post-hoc attributions: MASTER's gate α(m) (feature level), FactorVAE's α + β·z, HIST's three-way split, FinFormer's fusion gate. Agreement between intrinsic and post-hoc explanations is itself a result.
- **Comparisons across the two groups** must happen at the family / horizon level (Alpha158 names vs Alpha360 series × lag).
- **Before trusting any FactorVAE or Alpha360 number**, resolve issues 1–4. Otherwise differences between models may reflect implementation details rather than architecture, which is exactly the confound the thesis is trying to separate.
