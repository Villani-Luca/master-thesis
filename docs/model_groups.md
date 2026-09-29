# FinBench Models Grouped by Data & Experimental Setup

I've analyzed all the models inside FinBench by checking their data loaders and training scripts, and grouped them by the exact data sets and features they use.

Here is a summary of the 5 main groups I established for fair comparison:

- Group 1: Basic Price & Volume (DGDNN, D-Va, StockMixer) – Standard historical price and volume data.
- Group 2: Alpha158 Dataset (MASTER, MATCC, FactorVAE) – Cross-sectional 158-dimensional quantitative factor data.
- Group 3: Alpha360 Dataset (HIST, DiscoverPLF, FinFormer) – Cross-sectional 360-dimensional quantitative factor data.
- Group 4: Technical Indicators & Moving Averages (RTGCN, SVAT, SAMBA, Adv-ALSTM, CNNPred, ESTIMATE) – Engineered features (like RSI, MACD) or simple rolling averages on top of OHLCV.
- Group 5: Graph, Relations & Alternative Data (THGNN, HGTAN, STHAN-SR, MAN-SF) – Advanced models utilizing knowledge graphs, sector matrices, or NLP embeddings.


The models in FinBench can be grouped into several categories based on the data features and graph structures they utilize. Models within the same group share similar input requirements and evaluation protocols, making them directly comparable.

## Group 1: Basic Price & Volume (OHLCV)
Models relying strictly on raw or adjusted Open, High, Low, Close, and Volume data.

| Model | Task | Features Used |
|---|---|---|
| **DGDNN** | Classification | Raw OHLCV (`open`, `high`, `low`, `close`, `volume`) |
| **D-Va** | Regression | Adjusted OHLCV (`adj_open`, `adj_high`, `adj_low`, `adj_close`, `volume`) |
| **StockMixer** | Regression | Adjusted OHLCV (`adj_open`, `adj_high`, `adj_low`, `adj_close`, `volume`) |

## Group 2: Alpha158 Dataset
Models using the standard 158-dimensional Alpha dataset extracted from daily price/volume data.

| Model | Task | Features Used |
|---|---|---|
| **MASTER** | Regression | `_alpha158.csv` |
| **MATCC** | Regression | `_alpha158.csv` |
| **FactorVAE** | Regression | `_alpha158.csv` |

## Group 3: Alpha360 Dataset
Models using the extended 360-dimensional Alpha dataset.

| Model | Task | Features Used |
|---|---|---|
| **HIST** | Regression | `_alpha360.csv` |
| **DiscoverPLF** | Regression | `_alpha360.csv` |
| **FinFormer** | Regression | `_alpha360.csv` |

## Group 4: Technical Indicators & Moving Averages
Models that rely on pre-computed or dynamically computed technical indicators (e.g., Moving Averages, RSI, MACD, etc.).

| Model | Task | Features Used |
|---|---|---|
| **RTGCN** | Ranking | Adjusted Close + Moving Averages (5, 10, 20) |
| **SVAT** | Ranking | Raw Close + Moving Averages (5, 10, 20, 30) |
| **SAMBA** | Regression | Pre-computed technical indicators (`_tech.csv`) |
| **Adv-ALSTM** | Classification | OHLCV + Pre-computed technical indicators (`_tech.csv`) |
| **CNNPred2D / 3D** | Classification | Pre-computed tech indicators (`_tech.csv`) + Macro market data (`cnnpred_market.csv`) |
| **ESTIMATE** | Regression | Dynamically computed complex technicals (RSI, MACD, BB, ATR, MA) from raw `data.csv` |

## Group 5: Graph, Relations & Alternative Data
Models that require structural inputs (adjacency/incidence matrices) or external alternative data (News).

| Model | Task | Features Used | Graph / Alternative Data |
|---|---|---|---|
| **THGNN** | Classification | Raw OHLCV | Positive/Negative Correlation Adjacency Matrix |
| **HGTAN** | Classification | Raw OHLCV + Moving Averages | Sector/Industry Adjacency Matrix + Fund Matrix |
| **STHAN-SR** | Ranking | Adjusted Close + Moving Averages | Sector/Industry Incidence Matrix (`_inc_matrix_sect_ind.npz`) |
| **MAN-SF** | Classification | High, Low, Adjusted Close | Wikidata Adjacency Matrix + News Embeddings |
