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

## Aggiornamento del 5/11 sera: XAI su tutte le date

Le spiegazioni ora coprono **tutte** le date di ribilanciamento (PIANO.md, registro del 5/11). Le uscite a date campionate
sono in `code/results/xai_sampled_dates/`; tutto ciò che le usava va rifatto quando il core è rispiegato:

- [ ] **R1. Core su questo PC**: `grid.py run --xai --universes dji nasdaq100 sx5e --workers 5` (lanciato alle 18:26,
  `logs/xai_core.out`; avanzamento: `python code/scripts/grid.py status --xai --universes dji nasdaq100 sx5e`). Prima
  L = 5 (252 run), poi L = 1 e 20 (504). Su Windows FinFormer esce con 0xC0000409 a fine processo: il task risulta FAILED
  nel log ma le uscite sono complete (lo conferma `status --xai`); rilanciare lo stesso comando rifà solo ciò che manca.
- [ ] **R2. Shapley e LIME dji 2020** con tutte le date (`logs/xai_c2.out`, 18 run).
- [ ] **R3. Quando L = 5 è finito**: notebook 05, 06, 07, 08 rieseguiti; Observations e diagnosi ricontrollate;
  `thesis_figures.py res_inputs res_agreement res_agreement_models res_variance`; numeri dei cap. 6 e 9 aggiornati.
- [ ] **R4. Verifica del campionamento** (una frase nel cap. 5): classifiche con tutte le date contro una su due / una su
  quattro, per ogni run del core (`xai/` contro `xai_sampled_dates/`).
- [ ] **R5. Quando L = 1 e 20 sono finiti**: notebook 07 e 08 con il fattore orizzonte (E4).

Le voci C1, C2, C4, C4b, C5 qui sotto restano spuntate per la versione a date campionate; R1–R5 le sostituiscono.

## A. Quando finisce il riaddestramento (1242/1242)

Stima: domani 5/11, pomeriggio o sera.

- [x] **A1. Verifica della riproduzione** → 5/11 11:45: 99/117 identiche; le 18 diverse sono tutte e sole le MATCC
  (RankIC ±0,008 in media, max 0,024; MSE ±0,016; un cambio di segno su 18). Registro di PIANO.md.
  `python code/scripts/check_reproduction.py`
  Atteso: `117/117 identical`, **tranne le run MATCC** (non deterministiche anche sulla stessa GPU, vedi registro
  2026-11-05): per MATCC registrare quante differiscono e di quanto, e usare le nuove. Se qualche run è "different", **fermarsi**: i numeri dei seed 42 già nei
  notebook e nella tesi andrebbero rivisti (scrivere nel registro di PIANO.md).
- [x] **A2. Portafogli delle run riaddestrate** (5/11: L = 5 core ricalcolati con k di default e k = 5; 432
  portafogli con parquet, nessun rendimento giornaliero > 50%)
  ```
  python code/scripts/run_portfolio.py
  python code/scripts/run_portfolio.py --k 5 --universes dji nasdaq100 sx5e
  python code/scripts/run_portfolio.py --universes dji nasdaq100 sx5e --force
  ```
  Il terzo comando serve ai 18 portafogli seed 42 a L = 5 che hanno solo i CSV (senza parquet): li
  ricalcola uguali. Se si vuole limitare il `--force` al solo L = 5, va bene anche rifare tutto: dji e sx5e
  non cambiano con la regola delle aperture, nasdaq100 è già ricalcolato.
  Controllo: nessun portafoglio con rendimenti giornalieri > 50% (il controllo fatto il 4/11 è in PIANO.md).
- [x] **A3. Notebook 03** (5/11: rieseguito, 1242 run in `metrics_summary.csv`, Observations riscritte) (`code/notebooks/03_rolling_training.ipynb`): rieseguire tutto. Rigenera
  `code/results/analysis/metrics_summary.csv` con tutte le 1242 run. Riscrivere le **Observations**: i seed 0/5
  a L = 5 sono quelli della H200 (diversi dagli originali persi), più la nuova sezione 7 sugli orizzonti.
- [x] **A4. Notebook 04** (5/11: rieseguito, Observations riscritte) (`04_portfolio_baseline.ipynb`): rieseguire tutto (rigenera `portfolio_summary.csv`,
  `portfolio_margins.csv`, `portfolio_horizons.csv`). Riscrivere le **Observations** (nuova sezione 6 sugli
  orizzonti; a L = 1 portafogli ed equal weight perdono il rendimento notturno, vedi registro).
- [x] **A4b. Notebook 09** (5/11: rieseguito con i tre seed, Observations scritte) (`09_decision_fragility.ipynb`, Step 9.1): rieseguire con i tre seed (salva
  `results/analysis/decision_fragility.csv`) e scrivere le **Observations**.
- [x] **A5. Confronto con FinBench** (5/11: tabelle principali a L = 5 e complete per l'appendice `*_horizons.tex`;
  portafogli solo sugli universi della Table 7): `python code/scripts/compare_finbench.py` (dopo A3, perché legge
  `metrics_summary.csv`). Scrive `latex-thesis/tables/finbench_regression.tex` e `finbench_portfolio.tex`
  con i tre orizzonti.
- [x] **A6. Tesi, `04_framework.tex`** (5/11: paragrafi su training, riproducibilità, confronto FinBench su tre
  orizzonti e Table 7, riassunto; nuova appendice `98_appendix.tex` con le tabelle per orizzonte):
  - riscrivere il paragrafo "252 runs ... 88 hours on a single RTX 5090": 117 run sulla RTX 5090, 1125 sulla
    H200 NVL (~12 giorni, ~919 ore-GPU sommate sulle run); training riproducibile sulla stessa GPU (A1), non tra
    GPU diverse (i seed 0/5 rifatti sulla H200);
  - aggiornare il testo sulla Table finbench_regression ("Of the 18 model--universe pairs ..."), ora con tre
    orizzonti;
  - chiudere il TODO sul confronto con la Table 7 (`finbench_portfolio.tex`).
- [x] **A7. Tesi, cap. 7 (portafogli)** (5/11: protocollo e sezione dei risultati con le tabelle `pf_performance`
  e `pf_fragility` da `thesis_figures.py`): regola sulle aperture inaffidabili (SBNY) e l'effetto apertura a L = 1.

## B. Quando finiscono i 3 processi del batch XAI

Stima: tra stanotte (processo c) e domani sera (processo a). Non serve aspettare A.

Stato: processo c (FactorVAE, MATCC, MASTER) finito il 4/11 alle ~23:00; per i suoi modelli B1 e B2 sono già
fatti (`--reaggregate --models FactorVAE MATCC MASTER`, `--methods intrinsic --models MATCC MASTER`). Restano
DiscoverPLF (processo a). Processo b (FinFormer, HIST) finito il 5/11 alle ~07:40: B1 fatto
(`--reaggregate --models FinFormer HIST`), B2 fatto (pesi interni di 28 run FinFormer).

- [x] **B1. Ricostruire i file globali** (aggiunge la colonna `net`, che i processi partiti il 4/11 non scrivono):
  `python code/scripts/explain.py --reaggregate`
- [x] **B2. Pesi interni** di MASTER, MATCC, FinFormer (pochi secondi a run):
  `python code/scripts/explain.py --methods intrinsic`

## C. Quando A e B sono finiti

Stima: 6/11 sera – 7/11 mattina.

- [x] **C1. XAI sulle run riaddestrate** (seed 42 e i MASTER a L = 5), in 3 processi come il primo batch:
  ```
  python code/scripts/explain.py --models DiscoverPLF
  python code/scripts/explain.py --models FinFormer HIST
  python code/scripts/explain.py --models FactorVAE MATCC MASTER
  ```
  (salta da solo ciò che è già fatto; ora calcola anche `intrinsic` e la colonna `net`). ~12–18 ore.
- [ ] **C2. Shapley e LIME** (Step 6, controllo di IG), DJI 2020, tutti i modelli:
  `python code/scripts/explain.py --methods shapley lime --universes dji --years 2020`
- [x] **C3. Notebook 05** (5/11: rieseguito su MASTER dji seed 42 y2020, Observations scritte) (`05_xai_single_model.ipynb`, già impostato su MASTER dji L = 5 seed 42 y2020):
  rieseguire e scrivere le **Observations**.
- [ ] **C4. Notebook 06** (Step 6): rieseguire; scegliere e motivare il metodo principale e la misura per
  famiglia (`importance` lorda o `net`).
- [x] **C4b. Notebook 08** (5/11: rieseguito, Observations scritte; accordo per feature senza le feature di mercato) (Step 8, anni e regimi): rieseguire (serve B1 per le quote nette) e scrivere le
  **Observations**.
- [ ] **C5. Notebook 07** (5/11: diagnosi scritta; rieseguito alle 17:45 con xai2_a finito, 252/252 run complete; resta da rieseguire dopo il notebook 06 solo se cambiano i numeri del sanity check) (Step 7): rieseguire; scrivere la **diagnosi** dati vs modello. Tenere conto del
  sanity check (HIST e in parte MASTER hanno attribuzioni simili a quelle del modello non allenato).
- [ ] **C6. Tesi** (5/11: cap. 6 §6.2–6.7 scritti, tabelle `res_inputs`, `res_agreement`, `res_agreement_models`, `res_variance`; cap. 9 apertura e risposte alle RQ; introduzione: risposte in breve e contributo 2. Restano: cap. 5 valutazione e numeri `% TODO` dopo il notebook 06, i due `% TODO` sul sanity check nel cap. 6): sostituire i numeri marcati `% TODO` nei capitoli (`grep -n "TODO" latex-thesis/chapters/*.tex`),
  in particolare `05_xai_methodology.tex`; scrivere la sezione di valutazione (Step 6) e i risultati (cap. 6).

## E. XAI sugli universi grandi (S&P 500, STOXX Europe 600), tutti gli orizzonti: sulla H200

Decisione del 2026-11-05 (PIANO.md, registro): XAI su 5 universi × 3 orizzonti. I tre universi core su questo PC
(processi `xai3_*`), i due grandi sulla H200 del server del training (486 run). I processi `xai4_*` lanciati qui alle
17:55 sono stati fermati alle 18:00 prima di finire un run: nessun file scritto.

- [x] **E1. Pacchetto** (5/11): `C:\Users\villa\Downloads\xaifin_xai_bundle.zip` (1,4 GB: codice, i 486 checkpoint,
  il run di riferimento MASTER dji 2020 per `check_xai.py`, guida `START_HERE.md` = `code/scripts/remote/START_HERE_XAI.md`)
  e `xaifin_xai_data.zip` (dati di sp500, sxxp, dji, constituents; non serve se il server ha ancora i dati del training).
  Ricostruirli con i due comandi nel docstring di `code/scripts/remote/make_bundle.py`.
- [ ] **E2. Sul server**: Claude segue `START_HERE.md` (ambiente, `check_xai.py`, prova e misura, lancio di
  `xai.sh`, cioè `grid.py run --xai --universes sp500 sxxp`, tutte le date). Farsi dare le misure del passo 6 (secondi per run, RAM)
  e aggiornare la stima in PIANO.md.
- [ ] **E3. Ricezione** ogni 2–3 giorni del pacchetto leggero (`xaifin_xai_light_*.tar.gz`, senza tensori delle
  attribuzioni), alla fine di quello completo: estrarre dalla radice del repo (`tar -xzf xaifin_xai_light_<...>.tar.gz`).
  Scrive solo in `code/results/xai/*/*/{sp500,sxxp}/` e `logs/xai_server/`. Controllo:
  `python code/scripts/grid.py status --xai`; leggere `logs/xai_server/xai_check.txt`.
- [ ] **E4. Analisi**: notebook 07 e 08 con tutti gli universi e gli orizzonti (fattori universo e orizzonte nella
  scomposizione della varianza), poi capitoli 6 e 9 e le tabelle `res_*` di `thesis_figures.py` (ora fissate a L = 5
  e universi core).

## D. Alla fine

- [ ] **D1. Copia sul drive** di `code/results/Regression/`, `code/results/portfolio/` e `code/results/xai/`
  (i file grandi sono fuori da git: checkpoint, pickle, predizioni, holdings, attribuzioni).
- [ ] **D2. Commit** (i file piccoli: metriche, config, storie del training, global XAI, tabelle, notebook, tesi).
- [ ] **D3.** Aggiornare le voci di PIANO.md (D2, B6, B7, C1, C2) e cancellare questo file, o archiviarlo.
