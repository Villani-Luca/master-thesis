# Cosa rieseguire e quando

Promemoria operativo per chiudere il rientro della griglia e lo Step 5, scritto il 2026-11-04 mentre girano
il riaddestramento delle 117 run e il batch XAI. Il perché di ogni passo è nel registro di [PIANO.md](PIANO.md).
Spunta le voci (`- [x]`) man mano.

Tutti i comandi vanno lanciati dalla radice del repository, con l'ambiente attivo (`.venv\Scripts\activate`).

## 0. I job in corso e come capire se sono finiti

| Job | Log | Finito quando |
|---|---|---|
| Riaddestramento delle 117 run pre-griglia (RTX 5090) | `logs/grid_retrain_117.out`, `logs/grid/task*.log` | `python code/scripts/grid.py status` dice **1242/1242** |
| Batch XAI, seed 0/5 a L = 5 (3 processi) | `logs/xai_a.out`, `logs/xai_b.out`, `logs/xai_c.out` | l'ultima riga di ciascuno è `[N/N] ...` (a: 28, b: 56, c: 51 run) e il processo python non c'è più |
| Ricalcolo portafogli (regola sulle aperture) | `logs/portfolio_fix.out` | ✅ finito il 2026-11-04 ("ALL DONE") |

Errori: `grep -l Traceback logs/grid/*.log` e `grep Traceback logs/xai_*.err`.

---

## A. Quando finisce il riaddestramento (1242/1242)

Stima: domani 5/11, pomeriggio o sera.

- [ ] **A1. Verifica della riproduzione**
  `python code/scripts/check_reproduction.py`
  Atteso: `117/117 identical`. Se qualche run è "different", **fermarsi**: i numeri dei seed 42 già nei
  notebook e nella tesi andrebbero rivisti (scrivere nel registro di PIANO.md).
- [ ] **A2. Portafogli delle run riaddestrate**
  ```
  python code/scripts/run_portfolio.py
  python code/scripts/run_portfolio.py --k 5 --universes dji nasdaq100 sx5e
  python code/scripts/run_portfolio.py --universes dji nasdaq100 sx5e --force
  ```
  Il terzo comando serve ai 18 portafogli seed 42 a L = 5 che hanno solo i CSV (senza parquet): li
  ricalcola uguali. Se si vuole limitare il `--force` al solo L = 5, va bene anche rifare tutto: dji e sx5e
  non cambiano con la regola delle aperture, nasdaq100 è già ricalcolato.
  Controllo: nessun portafoglio con rendimenti giornalieri > 50% (il controllo fatto il 4/11 è in PIANO.md).
- [ ] **A3. Notebook 03** (`code/notebooks/03_rolling_training.ipynb`): rieseguire tutto. Rigenera
  `code/results/analysis/metrics_summary.csv` con tutte le 1242 run. Riscrivere le **Observations**: i seed 0/5
  a L = 5 sono quelli della H200 (diversi dagli originali persi), più la nuova sezione 7 sugli orizzonti.
- [ ] **A4. Notebook 04** (`04_portfolio_baseline.ipynb`): rieseguire tutto (rigenera `portfolio_summary.csv`,
  `portfolio_margins.csv`, `portfolio_horizons.csv`). Riscrivere le **Observations** (nuova sezione 6 sugli
  orizzonti; a L = 1 portafogli ed equal weight perdono il rendimento notturno, vedi registro).
- [ ] **A4b. Notebook 09** (`09_decision_fragility.ipynb`, Step 9.1): rieseguire con i tre seed (salva
  `results/analysis/decision_fragility.csv`) e scrivere le **Observations**.
- [ ] **A5. Confronto con FinBench**: `python code/scripts/compare_finbench.py` (dopo A3, perché legge
  `metrics_summary.csv`). Scrive `latex-thesis/tables/finbench_regression.tex` e `finbench_portfolio.tex`
  con i tre orizzonti.
- [ ] **A6. Tesi, `04_framework.tex`**:
  - riscrivere il paragrafo "252 runs ... 88 hours on a single RTX 5090": 117 run sulla RTX 5090, 1125 sulla
    H200 NVL (~12 giorni, ~919 ore-GPU sommate sulle run); training riproducibile sulla stessa GPU (A1), non tra
    GPU diverse (i seed 0/5 rifatti sulla H200);
  - aggiornare il testo sulla Table finbench_regression ("Of the 18 model--universe pairs ..."), ora con tre
    orizzonti;
  - chiudere il TODO sul confronto con la Table 7 (`finbench_portfolio.tex`).
- [ ] **A7. Tesi, cap. 7 (portafogli)**: regola sulle aperture inaffidabili (SBNY) e l'effetto apertura a L = 1.

## B. Quando finiscono i 3 processi del batch XAI

Stima: tra stanotte (processo c) e domani sera (processo a). Non serve aspettare A.

Stato: processo c (FactorVAE, MATCC, MASTER) finito il 4/11 alle ~23:00; per i suoi modelli B1 e B2 sono già
fatti (`--reaggregate --models FactorVAE MATCC MASTER`, `--methods intrinsic --models MATCC MASTER`). Restano
DiscoverPLF (processo a), FinFormer e HIST (processo b).

- [ ] **B1. Ricostruire i file globali** (aggiunge la colonna `net`, che i processi partiti il 4/11 non scrivono):
  `python code/scripts/explain.py --reaggregate`
- [ ] **B2. Pesi interni** di MASTER, MATCC, FinFormer (pochi secondi a run):
  `python code/scripts/explain.py --methods intrinsic`

## C. Quando A e B sono finiti

Stima: 6/11 sera – 7/11 mattina.

- [ ] **C1. XAI sulle run riaddestrate** (seed 42 e i MASTER a L = 5), in 3 processi come il primo batch:
  ```
  python code/scripts/explain.py --models DiscoverPLF
  python code/scripts/explain.py --models FinFormer HIST
  python code/scripts/explain.py --models FactorVAE MATCC MASTER
  ```
  (salta da solo ciò che è già fatto; ora calcola anche `intrinsic` e la colonna `net`). ~12–18 ore.
- [ ] **C2. Shapley e LIME** (Step 6, controllo di IG), DJI 2020, tutti i modelli:
  `python code/scripts/explain.py --methods shapley lime --universes dji --years 2020`
- [ ] **C3. Notebook 05** (`05_xai_single_model.ipynb`, già impostato su MASTER dji L = 5 seed 42 y2020):
  rieseguire e scrivere le **Observations**.
- [ ] **C4. Notebook 06** (Step 6): rieseguire; scegliere e motivare il metodo principale e la misura per
  famiglia (`importance` lorda o `net`).
- [ ] **C4b. Notebook 08** (Step 8, anni e regimi): rieseguire (serve B1 per le quote nette) e scrivere le
  **Observations**.
- [ ] **C5. Notebook 07** (Step 7): rieseguire; scrivere la **diagnosi** dati vs modello. Tenere conto del
  sanity check (HIST e in parte MASTER hanno attribuzioni simili a quelle del modello non allenato).
- [ ] **C6. Tesi**: sostituire i numeri marcati `% TODO` nei capitoli (`grep -n "TODO" latex-thesis/chapters/*.tex`),
  in particolare `05_xai_methodology.tex`; scrivere la sezione di valutazione (Step 6) e i risultati (cap. 6).

## D. Alla fine

- [ ] **D1. Copia sul drive** di `code/results/Regression/`, `code/results/portfolio/` e `code/results/xai/`
  (i file grandi sono fuori da git: checkpoint, pickle, predizioni, holdings, attribuzioni).
- [ ] **D2. Commit** (i file piccoli: metriche, config, storie del training, global XAI, tabelle, notebook, tesi).
- [ ] **D3.** Aggiornare le voci di PIANO.md (D2, B6, B7, C1, C2) e cancellare questo file, o archiviarlo.
