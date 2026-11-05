# Piano di lavoro fino al 18/11/2026

Documento di progetto per il periodo in cui la griglia di training gira sulla H200 dell'amico.
Il lavoro descritto qui si fa sulla **macchina con CUDA** (quella degli stage 1–4).
Il piano tecnico completo resta [THESIS_GUIDE.md](THESIS_GUIDE.md); qui ci sono lo stato reale,
cosa fare, le decisioni prese e il registro. Aggiornato il 2026-10-22.

## Per Claude (o chi riprende il lavoro)

1. Leggi questo file, poi le parti della guida citate (Step 5–9, 11), poi `docs/model_notes.md`
   per i dettagli dei modelli.
2. Fai le voci della sezione "Lavoro" **in ordine**, spuntandole qui (`- [x]`) con una riga su cosa
   è stato fatto. Le decisioni nuove vanno nel registro in fondo, con la data.
3. Regole:
   - non modificare modelli, iperparametri, seed, protocollo o regole di selezione: lo XAI spiega i
     checkpoint così come sono;
   - non cancellare e non sovrascrivere nulla in `code/results/Regression/` (`--force` mai);
   - lo XAI scrive solo nelle sue cartelle (§2.2 della guida);
   - le figure per la tesi vanno in `latex-thesis/images/results/` con nomi fissi;
   - la tesi è **in inglese**; codice e commenti nello stile del repo.
4. Se una scelta non è coperta da questo file o dalla guida e cambia i risultati, chiedi prima.

## Obiettivo e vincoli

- **Scadenza: tesi completa (codice, analisi, scrittura) entro il 18/11/2026. Bozza al relatore
  verso il 10/11.**
- Domanda di ricerca: come le feature guidano le predizioni e l'allocazione top-k, e se
  l'instabilità viene dai **dati** o dal **design dei modelli** (Step 7: è il risultato centrale e
  non si taglia).
- Tagliati per la scadenza (sviluppi futuri nella tesi): SHAP/LIME oltre dji y2020, Step 9.2,
  Step 10, pagina app `5_Over_time`, riferimento LightGBM/TreeSHAP (salvo tempo avanzato).
- Orizzonti L = 1, 5, 20 **obbligatori** per tutti e 5 gli universi (li produce la griglia H200).

## Stato reale (verificato il 2026-10-22)

| Cosa | Stato |
|---|---|
| Griglia sulla H200 | avviata dall'amico con `launch_grid.py` (orchestratore scritto dal suo Claude): 1242 run, L = 1/5/20, 5 universi, 3 seed. ETA ~13 giorni (12–17) → rientro **~4–9 novembre**. Stato sul suo server: `grid.py status` e `_job/claude/status.json`. Risultati con `code/scripts/remote/pack_results.sh` |
| Run finite disponibili | **117** (L = 5): seed 42 per tutti e 6 i modelli su dji, nasdaq100, sx5e (84 run), MASTER anche con seed 0 e 5, MATCC e FactorVAE con qualche seed 0/5 sparso. **Le run seed 0/5 dello stage 4 sono perse**: arriveranno dalla griglia H200 |
| Checkpoint | `model.pth` è in `.gitignore`: non passa con git. Sulla macchina CUDA verificare con `grid.py status` che ci siano le 117 run; se mancano, trasferire lo zip di `export_results.py --zip` |
| Ambiente | la guida §0.1 dà captum, shap e lime installati sulla macchina CUDA: verificarlo con `00_environment_check.ipynb` (il PC Windows ha torch solo CPU e non li ha) |
| Tesi (`latex-thesis/`, in inglese) | cap. 2 Background ~3700 parole (bozza); introduzione ~60; cap. 3 è un segnaposto ("Tables and Images"); conclusioni vuote; mancano metodologia e risultati |
| Codice | Step 0–4 fatti; `xai/`, `analysis/`, `perturb/` vuoti; app con `Home`, `1_Data`, `2_Models`, `6_Portfolio` |

## Decisioni prese (2026-10-22)

| # | Domanda | Decisione |
|---|---|---|
| D1 | Run seed 0/5 dello stage 4 | perse. XAI subito sulle 117 run; il confronto tra seed (Step 7) usa MASTER finché non arrivano i seed 0/5 dalla H200, poi si estende |
| D2 | Dove gira lo XAI | macchina con CUDA (sviluppo e batch) |
| D3 | Metodi | Integrated Gradients (+ Saliency) su tutte le run; permutation importance cross-sectional; attention dove il modello la espone facilmente; SHAP/LIME solo su dji y2020, tutti i modelli, come controllo di IG |
| D4 | Cosa si spiega | sia il punteggio (`target="pred"`) sia l'appartenenza al top-k (`target="topk_margin"`) |
| D5 | Granularità di permutation e SHAP/LIME | le 8 **famiglie** di `xaifin.data.features.FAMILIES` (IG resta a risoluzione piena T × F e si aggrega dopo) |
| D6 | Bozza al relatore | verso il 10/11 |

## Lavoro

In ordine. Ogni voce ha il criterio di "fatto".

### A. Avvio sulla macchina CUDA
- [ ] **A1. Codice e dati**: repo aggiornato (commit/push dal PC Windows, pull qui), `pip install -e code`,
  `grid.py status` mostra le 117 run (altrimenti trasferire i checkpoint).
- [ ] **A2. Ambiente**: `00_environment_check.ipynb` gira senza errori (CUDA, captum, shap, lime).

### B. Step 5: motore XAI
- [x] **B1. Misura del costo** di IG (50 passi, entrambi i target) su MASTER e FactorVAE, dji y2020:
  secondi per data, memoria GPU. Stima del batch sulle 117 run e sulle run seed 0/5 future.
  *Fatto quando:* c'è un numero in ore, scritto nel registro.
  → 2026-11-04, su tutti e sei i modelli (dji e nasdaq100): vedi registro ("B1, costo dello XAI").
- [x] **B2. `xai/base.py`**: protocollo `Explainer`, `explain(adapter, batch, target) -> attr[N,T,F]`
  con `target` in {`pred`, `topk_margin`}; date di ribilanciamento di L = 5 (le stesse per ogni
  orizzonte); baseline zero; salvataggio float16 nel results store.
  → 2026-11-04: `target_scores`, `own_gradients` (blocchi diagonali dello Jacobiano), `explain_dates`
  (una data di ribilanciamento su due), `explain_run`, `save_attributions`; baseline media del giorno
  (decisione nel registro).
- [x] **B3. Metodi** in `xai/`: IG e Saliency (captum), permutation importance per famiglia
  → 2026-11-04: IG e Saliency (`xai/gradient.py`, nodi Gauss-Legendre di Captum; Captum non si usa
  direttamente perché espanderebbe la dimensione dei titoli) e permutation per famiglia
  (`xai/permutation.py`) fatti. Attention di MASTER fatta (`xai/intrinsic.py`: feature gate, attention
  temporale, intra- e inter-titolo, ricostruite dal forward senza toccare il modello, differenza 0).
  2026-11-04: anche MATCC (filtro di mercato additivo, attention tra titoli e temporale) e FinFormer (gate di
  fusione spaziale/temporale, attention dinamica e statica tra titoli, attention sui 60 "passi" = 6 serie per
  la issue 4), con hook in sola lettura, differenza 0. B3 chiuso; HIST e FactorVAE senza pesi interni.
  (mescola la famiglia tra i titoli dello stesso giorno, misura ΔRankIC e cambio del top-k),
  attention. Prima tutto su MASTER × dji × y2020, poi gli altri cinque modelli. Casi delicati: grafo
  di HIST, FactorVAE, attention da far restituire. **Al massimo un giorno ciascuno**: se non basta,
  quel modello resta senza attention (annotarlo nel registro).
- [ ] **B4. SHAP/LIME** a gruppi (famiglie) su dji y2020, tutti i modelli.
  → 2026-11-04: `xai/shap_lime.py`: Shapley **esatto** sulle famiglie (2^7 = 128 coalizioni per giorno,
  famiglia assente = titolo medio del giorno per tutti i titoli; efficienza esatta, 1e-8) e LIME sulla stessa
  rappresentazione; `explain.py --methods shapley lime`. Prova (MASTER dji L = 1): ~1,7 s/giorno, LIME vs
  Shapley 0,88–0,92 per titolo; quote nette di IG per famiglia ≈ Shapley (scarti ≤ 0,01). Da lanciare su
  dji y2020 seed 42, tutti i modelli, a riaddestramento finito.
- [x] **B5. `xai/aggregate.py`**: |a| normalizzato per campione, importanza per feature, lag,
  famiglia; segno medio; dispersione; ranking → `global.parquet` per run e metodo.
  → 2026-11-04: anche per fonte e orizzonte; `global_<target>.parquet` per run e metodo.
- [ ] **B6. `scripts/explain.py`** (resumable come `train_rolling.py`) e batch sulle 117 run.
  → 2026-11-04: script fatto. Batch partito sulle run L = 5 con checkpoint (seed 0/5 dalla H200) in 3
  processi staccati (`logs/xai_a|b|c.out`); il seed 42 e i MASTER seguono a training finito.
- [ ] **B7. Notebook `05_xai_single_model.ipynb`** con osservazioni.
  → 2026-11-04: notebook scritto (6 sezioni), collaudato su MASTER dji L = 1 seed 5 y2022; da eseguire su
  MASTER dji L = 5 seed 42 y2020 a riaddestramento finito e scrivere le osservazioni. Prime indicazioni
  (run di prova): IG su `pred` e `topk_margin` quasi identici (Spearman 0,9998); permutation: il ranking
  delle famiglie per cambio del top-k concorda con IG (0,93), quello per ΔRankIC no (0,43–0,57); own vs
  joint 0,998, baseline zero vs media 0,84, completezza mediana 0,5%; gate di MASTER vs IG 0,21.
- *Fatto quando:* `global.parquet` per IG e permutation su tutte le 117 run, per entrambi i target.

### C. Step 6–9 sulle attribuzioni
- [ ] **C1. Step 6 ridotto**: sanity check a pesi randomizzati, accordo tra metodi (IG, permutation,
  SHAP, LIME su dji y2020), deletion curve → notebook 06 con la scelta motivata dei metodi.
  → 2026-11-04: `xai/evaluation.py` (sanity check contro i pesi iniziali, deletion curve "joint" e "own",
  AOPC) scritto e provato su MASTER, HIST, FactorVAE (prime evidenze nel registro). Notebook
  `06_xai_evaluation.ipynb` scritto e collaudato (dji seed 5 y2022, 5 modelli; MASTER manca finché non è
  riaddestrato): vedi registro. Mancano l'esecuzione sul seed 42 e la scelta motivata.
- [ ] **C2. Step 7 completo**: `analysis/agreement.py` (Spearman, Kendall, Jaccard top-k, RBO),
  `analysis/variance.py` (fattori: modello, anno, universo; seed per MASTER subito, per tutti dopo il
  rientro; orizzonte se arrivano in tempo le attribuzioni L = 1, 20), confronto `pred` vs
  `topk_margin`, diagnosi dati vs modello scritta, notebook 07.
  → 2026-11-04: `analysis/agreement.py` (Spearman, Kendall, Jaccard top-k, RBO esteso) e
  `analysis/variance.py` (`load_importance`, `pair_agreement` sulle coppie che differiscono di un solo
  fattore, `decomposition` ANOVA tipo II: modello, universo, anno, interazioni, seed) scritti e collaudati su
  dati sintetici. Nota: su 8 famiglie l'RBO resta alto anche per ranking opposti (0,74): a livello di
  famiglia leggere Spearman/Kendall. Notebook `07_cross_model_comparison.ipynb` scritto (accordo per fattore,
  modello × modello, scomposizione della varianza, feature dominanti, `pred` vs `topk_margin` e ΔRankIC vs
  cambio del top-k, accordo contro RankIC) e collaudato sulle prime run spiegate: si riesegue così com'è
  quando il batch è completo. Manca la diagnosi scritta.
- [ ] **C3. Step 8 leggero**: per anno, 2020 e 2022 contro gli anni calmi; IC, rendimenti, quote
  delle famiglie, deriva anno su anno; notebook 08.
  → 2026-11-04: notebook `08_periods_and_regimes.ipynb` scritto e collaudato (regimi del cap. 3: 2020 e 2022
  alta volatilità, 2021/2023/2024 calmi): RankIC per anno e regime contro la volatilità dell'indice,
  rendimento in eccesso sull'equal weight per anno, quote nette delle famiglie per anno e differenza tra
  regimi, deriva anno su anno delle spiegazioni (anni consecutivi, transizioni calme vs con anno volatile).
  Da rieseguire a batch XAI completo e dopo `--reaggregate`; mancano le osservazioni.
- [ ] **C4. Step 9.1**: fragilità delle decisioni (margini dello Step 4 contro il rumore tra seed:
  subito su MASTER, poi su tutti).
  → 2026-11-04: `portfolio/fragility.py` (rumore tra seed = std dello z-score giornaliero del titolo tra i seed;
  posizione fragile se margin_z < rumore) e notebook `09_decision_fragility.ipynb`, provati sui seed 0 e 5:
  la quota di posizioni fragili coincide con il turnover osservato tra i portafogli di due seed (14 coppie
  modello × universo: correlazione 0,976, scarto medio 0,014). Posizioni fragili: 42–50% per FactorVAE,
  HIST, DiscoverPLF; 55–70% per FinFormer e MATCC. Da rieseguire con i tre seed.

### D. Preparazione al rientro della griglia
- [x] **D1. Notebook 03 e 04** generici su 5 universi × 3 orizzonti: rieseguibili senza modifiche.
  → 2026-11-04: sezioni 1-6 a L = 5 su tutti e 5 gli universi, nuova sezione "Horizons" (L = 1, 5, 20) in
  entrambi; il 04 legge solo i portafogli con il k di default (i top-5 servono a `compare_finbench.py`) e
  salva anche `portfolio_horizons.csv`.
- [ ] **D2. Procedura di rientro**: estrarre il pacchetto, `grid.py status` = 1242/1242,
  `export_results.py`, `run_portfolio.py`, notebook 03/04, `compare_finbench.py`, poi
  `explain.py` sulle nuove run L = 5 (seed 0/5) e, se c'è tempo, L = 1 e 20 su dji, nasdaq100, sx5e.
  → 2026-11-04: pacchetto (`xaifin_results_luca-gpu-priv-0_20261103_0809`) copiato, 1242/1242, nessun
  traceback nei log (in `logs/grid_h200/`); `export_results.py` fatto; portafogli fatti per le run con pickle.
  **In corso:** riaddestramento sulla RTX 5090 delle 117 run pre-griglia (vedi registro), poi
  `run_portfolio.py` (anche `--k 5`, `--force` sui 18 seed 42 senza parquet), notebook 03/04, `compare_finbench.py`.
  Processi staccati: `logs/grid_retrain_117.out`, `logs/portfolio_fix.out` (ricalcolo dopo la regola sulle
  aperture, vedi registro); finiti `logs/portfolio_k5.out`, `logs/portfolio_sxxp.out`.
  **Da fare, in ordine** (versione operativa con i comandi e i momenti: [RIESECUZIONE.md](RIESECUZIONE.md)):
  1. ✅ quando finiscono i portafogli: collaudare il notebook 04; capire perché l'equal weight di sxxp rende
     ~-2% l'anno contro +6% dell'indice (→ è l'effetto apertura a L = 1, vedi registro);
  2. quando `grid.py status` torna a 1242/1242: confrontare le 117 run rifatte con
     `results/legacy/pre_grid_rtx5090/` (attese identiche: stessa GPU, stesso torch) e scriverlo nel registro;
  3. `run_portfolio.py` e `run_portfolio.py --k 5` per i gruppi che mancano; `--force` sui 18 portafogli
     seed 42 L = 5 (hanno solo i CSV, senza parquet);
  4. rieseguire i notebook 03 e 04 (rigenerano `metrics_summary.csv`, `portfolio_*.csv`) e riscrivere le
     "Observations" (i seed 0/5 di L = 5 sono cambiati; nuove sezioni Horizons);
  5. `compare_finbench.py` (tabelle `finbench_regression.tex` e `finbench_portfolio.tex` con i tre orizzonti);
  6. tesi: TODO di `03_data.tex` (quali capitoli usano la griglia estesa) e di `04_framework.tex` (totali H200,
     paragrafo "252 runs / 88 hours" da riscrivere, confronto con la Table 7, testo sui 18 pair da aggiornare);
     dire che il training non è riproducibile tra GPU diverse;
  7. copiare `code/results/Regression/` e `code/results/portfolio/` sul drive (~6 GB, fuori da git);
  8. poi Step 5 (XAI): i checkpoint di tutte le 1242 run sono sul disco. Quando finiscono i 3 processi del
     batch XAI: `explain.py --reaggregate` (aggiunge la colonna `net` ai global già scritti); poi
     `explain.py --methods intrinsic` (pesi interni di MASTER, MATCC, FinFormer: pochi secondi a run, i processi già
     partiti non li calcolano); `explain.py` sul seed 42 e sui MASTER L = 5 (ora include `intrinsic`); `explain.py --methods shapley lime --universes dji
     --years 2020 --seeds 42`; notebook 05 sulla run vera e osservazioni.
- [ ] **D3. Pacchetti parziali**: chiedere all'amico un `pack_results.sh` ogni ~3 giorni, per far
  partire lo XAI sulle nuove run senza aspettare la fine.

### E. App (Step 11, ridotta)
- [ ] **E1. `3_Explain`**: ranking globale + famiglie, heatmap T × F per data e titolo, attention.
- [ ] **E2. `4_Compare`**: ranking affiancati, heatmap di accordo, dispersione tra seed.

### F. Scrittura (in inglese, nelle pause dei batch)
- [ ] **F1. Metodologia**: dati e universi, modelli, protocollo rolling, regole di selezione, portafoglio.
  → 2026-11-04: cap. 3 completo nelle figure (`data_labels`, `data_alpha158_corr`, `data_volatility` in
  `images/results/`, tabella `tables/data_label_std.tex`, tutte da `code/scripts/thesis_figures.py`, che
  esporta le figure di `xaifin.viz` in PDF con kaleido 0.1.0.post1) e nei due grid (core / estesa); cap. 7:
  scritto il protocollo del backtest (regola, rendimenti, equivalenza con FinBench, aperture inaffidabili,
  giorni con pochi titoli, effetto apertura a L = 1, benchmark e metriche, margini e overlap); i risultati
  aspettano il notebook 04 completo.
- [ ] **F2. Prestazioni e portafogli** (notebook 03/04; numeri aggiornati al rientro).
- [ ] **F3. Metodi XAI e loro valutazione.**
  → 2026-11-04: prima bozza di `05_xai_methodology.tex` (~2900 parole: cosa si spiega, metodi, baseline,
  percorso joint/own, aggregazione lorda/netta, costi, piano di valutazione). Numeri da run di prova marcati
  `% TODO`; manca la sezione con i risultati dello Step 6. Allineate due frasi del cap. 2 (baseline, percorso).
  Non compilata: LaTeX non è installato su questo PC.
- [ ] **F4. Risultati XAI, diagnosi, conclusioni** (dopo C2).
- [ ] **F5. Introduzione** (alla fine); il cap. 3 segnaposto va sostituito.

## Calendario

| Date | Lavoro |
|---|---|
| 22–24 ott | A1, A2, B1, B2 su MASTER; F1 |
| 25 ott – 1 nov | B3–B7 (batch sulle 117 run); F3 |
| 2–6 nov | C1, C2 (seed solo MASTER); D1; F2 sui numeri attuali |
| ~4–9 nov | rientro griglia → D2; C2 esteso ai seed, C3, C4 |
| **~10 nov** | **bozza al relatore**: metodologia, prestazioni, primi risultati XAI |
| 10–17 nov | E1, E2; F4, F5; correzioni del relatore |
| 18 nov | consegna |

## Registro delle decisioni

| Data | Decisione |
|---|---|
| 2026-10-22 | Scadenza 18/11, bozza ~10/11; scope tagliato (anche in THESIS_GUIDE.md) |
| 2026-10-22 | L = 1 e L = 20 obbligatori; la griglia intera gira sulla H200 dell'amico con `launch_grid.py` |
| 2026-10-22 | `grid.py`/`train.sh` del repo aggiornati con tetto RAM, watchdog, THREADS = 2 (non usati dalla run in corso, che ha il suo orchestratore) |
| 2026-10-22 | Decisioni D1–D6 (tabella sopra) |
| 2026-11-04 | Griglia H200 rientrata: 1242/1242 (1125 run nuove), ~919 ore-GPU sommate sulle run, ~12 giorni (22/10–3/11) |
| 2026-11-04 | **Le run riaddestrate su H200 non coincidono con quelle della RTX 5090**: i seed 0/5 di L = 5 (persi, rifatti sulla H200) differiscono dagli originali in `metrics_summary.csv` del commit 34abdca (IC fino a 0,055; epoca tenuta diversa in 10–14 run su 14 per HIST, DiscoverPLF, FinFormer; FactorVAE quasi uguale). Stesso codice e stesso torch: è il non determinismo tra GPU diverse. Le osservazioni dei notebook 03/04 e il testo della tesi sui seed vanno aggiornati |
| 2026-11-04 | Le 117 run pre-griglia avevano solo le copie CSV (né `model.pth` né pickle su questo PC): spostate in `results/legacy/pre_grid_rtx5090/` e **riaddestrate sulla RTX 5090** (`grid.py run --horizons 5 --universes dji nasdaq100 sx5e`, stesso torch 2.11.0+cu128), confrontando poi le metriche con quelle vecchie |
| 2026-11-04 | Giorni con meno di k titoli (festività europee: sxxp ha 5–7 titoli su ~400 a Pasquetta e il 26/12): si tengono tutti gli n < k titoli a peso 1/n, come FinBench (`topk_holdings`); margine NaN in quel periodo |
| 2026-11-04 | **Aperture inaffidabili nel backtest** (`backtest.asset_returns`): il filtro qualità guarda solo le chiusure, ma il primo giorno il portafoglio compra all'apertura. SBNY.US (dopo il fallimento) apre a 0,0003 e chiude a 0,05 il 31/07/2023 (+16.000%): tutti i portafogli sp500 L = 1 avevano CAGR +100–256%. Regola: un'apertura sotto `MIN_PRICE` o con chiusura/apertura oltre `MAX_DAILY_MOVE` (o sotto 1/`MAX_DAILY_MOVE`) non si usa, quel giorno vale il rendimento close-to-close. Coppie (titolo, giorno) toccate nel 2020–2024: dji 0, sx5e 0 (confronto con FinBench invariato), nasdaq100 557 (VBVT.US a 0,0001), sp500 36 (SBNY.US, FTR.US), sxxp 63 (ORRON.ST). Portafogli di nasdaq100, sp500, sxxp ricalcolati con `--force` (`logs/portfolio_fix.out`). Da dire nella tesi (cap. 7, protocollo del backtest) |
| 2026-11-04 | **B1, costo dello XAI** (RTX 5090 condivisa con il training, quindi tempi per eccesso). `xai/base.py` e `xai/gradient.py` scritti (Saliency, IG con percorso "own" o "joint", baseline "zero" o "mean"). Quota del gradiente di s_i che passa per gli input degli altri titoli: MASTER 34%, HIST 69%, FactorVAE 1%. IG "own" (la definizione della guida) costa 4–10× "joint" e i due danno ranking quasi uguali (Spearman per titolo 0,986–1,000). **FactorVAE: baseline zero inadatta** (LayerNorm sulle feature in ingresso, f(a·x_i) = f(x_i): il percorso salta in 0 e IG non converge); con la baseline "media del giorno" converge già con 16 passi (Spearman 16 vs 64 passi ≥ 0,998, completezza entro ~2%). IG joint, 16 passi, per giorno e target: dji MASTER 0,7 s, MATCC 0,8, FactorVAE 7,7, HIST 9,7, DiscoverPLF 23, FinFormer 9; nasdaq100 ~3× (lineare in N). Batch completo L = 5 (252 run × 2 target) ≈ 117 ore-GPU in serie; Saliency ≈ 1/16 di IG |
| 2026-11-04 | **Decisioni XAI** (sostituiscono "baseline zero" di B2): IG con **baseline = input medio cross-sectional del giorno** per tutti i modelli (spiega perché il titolo ha un punteggio diverso dal titolo medio di quel giorno; la baseline zero resta come controllo dello Step 6 su una run); percorso **"joint"** (il percorso "own" della guida si calcola su qualche run come controllo e la differenza va riportata); 16 passi Gauss-Legendre; **tutti e tre i seed, una data di ribilanciamento settimanale su due** (~26 giorni l'anno, ~60 ore-GPU per IG su tutte le run L = 5 e i due target) |
| 2026-11-05 | Task 230, 239 e 257 del riaddestramento (FinFormer dji, nasdaq100, sx5e seed 42) segnati FAILED (exit 0xC0000409) tra le 08:13 e le 10:36: è un crash di Windows **alla chiusura** del processo, dopo il salvataggio dell'ultima run, sempre e solo per FinFormer; le run sono tutte complete e identiche alle originali. Nessuna azione |
| 2026-11-05 | **MATCC non è bit-riproducibile nemmeno sulla stessa GPU.** La prima run MATCC rifatta (dji seed 0 y2020) differisce dall'originale: stessa GPU, torch, Python ed epoca tenuta; epoca 0 identica, dall'epoca 1 scarti all'ultima cifra (loss 0,941244 vs 0,941250) che crescono in 70 epoche → test RankIC 0,012 vs 0,025, MSE 1,655 vs 1,621. FactorVAE e DiscoverPLF (22 run) restano bit-identici, quindi è un kernel CUDA non deterministico di MATCC (RWKV/scheduler; `use_deterministic_algorithms(warn_only=True)` avvisa soltanto), già visibile nel test di equivalenza (4e-5 dopo un'epoca). Probabilmente tutte le MATCC rifatte differiranno un po': i numeri di MATCC nei notebook e nel cap. 4 si rigenerano comunque (A3–A6); scriverlo nella tesi. Verificare sulle altre run MATCC |
| 2026-11-04 | **Riproduzione verificata** sulle prime 3 run rifatte sulla RTX 5090 (FactorVAE dji seed 0 e 42, nasdaq100 seed 42, y2020): metriche, predizioni (differenza 0,0) e `train_history.csv` identici alle originali di `legacy/pre_grid_rtx5090/`, tranne la colonna `seconds`. Conferma che le differenze dei seed 0/5 rifatti sulla H200 vengono dalla GPU diversa (non riproducibilità tra GPU, non un bug). Da ripetere su tutte le 117 a fine riaddestramento |
| 2026-11-04 | Pesi interni (run di prova dji seed 5 y2022): l'**attention tra titoli è quasi uniforme** in MASTER, MATCC e FinFormer (peso su sé stessi ≈ 1/N; solo l'attention statica di FinFormer, sul grafo dei settori, è concentrata: 6× l'uniforme). MATCC mette il 91% dell'attention temporale sull'ultimo giorno della finestra di 20; MASTER il ~80% sugli ultimi 2–3. FinFormer: 64% della rappresentazione dal ramo temporale; tra le serie LOW 25%, HIGH 20%, CLOSE 18%, OPEN 16%, VOLUME 12%, VWAP 9%. Da rifare sulle run finali (Step 5, notebook 05) |
| 2026-11-04 | **Step 6, notebook 06 di prova** (dji seed 5 y2022; sanity e deletion su 1 giorno, accordo tra metodi su tutta la run). *Deletion titolo per titolo*: IG fedele per tutti e 5 (calo 0,06–2,05, casuale ≈ 0); quella "joint" fallisce per HIST e DiscoverPLF → nella tesi usare quella titolo per titolo; Saliency mai fedele. *Sanity*: passano FactorVAE, DiscoverPLF, FinFormer (0,34–0,47); HIST no (0,99); MATCC in parte (0,72–0,80, IG dominato da \|x − b\|: 0,85). Spostamento dei pesi dall'inizializzazione: MATCC 97%, DiscoverPLF 57%, FinFormer 28%, gli altri ~5%. *Accordo per famiglia* (media): IG netto vs Shapley 0,88, LIME vs Shapley 0,99, IG vs permutation top-k 0,91–0,97; **permutation ΔRankIC non concorda con nessuno (0,1–0,18)**. *Percorso*: own vs joint 1,00 per MATCC, FactorVAE, HIST, 0,95 DiscoverPLF, **0,75 FinFormer**; baseline zero vs media 0,61–0,99 |
| 2026-11-04 | **Anomalia equal weight di sxxp spiegata: è l'orizzonte L = 1, in tutti gli universi.** CAGR medio dell'equal weight a L = 1 / 5 / 20: dji 1,3 / 5,9 / 6,1%, sp500 1,3 / 9,2 / 9,1%, sxxp −1,9 / 5,0 / 5,8% (indice 8,7, 13,4, 6,0%). Il protocollo FinBench compra all'apertura del primo giorno di ogni periodo: a L = 1 ogni giorno è un primo giorno, quindi portafogli e equal weight guadagnano solo il rendimento apertura-chiusura e perdono quello notturno (sxxp 2021–2024: intraday −1,5% l'anno, notturno +13,7%). Il confronto modello vs equal weight a L = 1 resta corretto (stesso svantaggio), quello con l'indice no. Protocollo FinBench mantenuto; da scrivere nel cap. 7 |
| 2026-11-04 | **Step 6, prime evidenze** (`xai/evaluation.py`; run di prova dji, 3 giorni). *Deletion curve*: IG fedele per MASTER (calo del target 0,14–0,59 togliendo l'1–50% degli input, casuale ≈ 0) e FactorVAE (0,08–0,51); per HIST solo togliendo un titolo alla volta (0,03–0,07; togliendo tutti i titoli insieme gli effetti attraverso i concetti si compensano): per i modelli molto cross-sectional la deletion va fatta titolo per titolo. Saliency non è fedele (≈ casuale). *Sanity check* (IG del modello allenato vs lo stesso con i pesi iniziali del suo seed): FactorVAE passa (Spearman 0,40); **MASTER 0,79–0,87 e HIST 0,99**. Il training sposta i pesi solo del ~5% in tutti e tre. MASTER: \|IG\| segue la deviazione dell'input dal titolo medio (Spearman 0,84), quindi le sue importanze vanno lette rispetto a questo riferimento; HIST: il profilo è fissato dall'architettura (GRU sulle colonne in ordine sbagliato), non dal training. Da rifare sulle run finali (notebook 06) e da tenere nella diagnosi dello Step 7 |
| 2026-11-04 | Effetti da riportare nella tesi (Step 5): (1) con la baseline media del giorno le feature di **mercato** (uguali per tutti i titoli) hanno attribuzione nulla per costruzione e la permutation non le tocca; il loro effetto passa dentro le altre famiglie (gate di MASTER/MATCC). (2) **HIST ignora OPEN0, HIGH0, LOW0, VWAP0** (gradiente ~1e-10, permutare la famiglia candle non cambia i punteggi): con l'ordine sbagliato delle colonne Alpha360 (issue 4) finiscono nei primi passi della GRU di 60 passi e il loro effetto svanisce; DiscoverPLF e FinFormer le usano |
| 2026-11-04 | Aggregazione per famiglia: due misure in `global_*.parquet`. `importance` = quota di Σ\|a\| (sensibilità agli input della famiglia, conta anche input che spingono in direzioni opposte); `net` = quota di \|Σa\| (contributo totale della famiglia, ciò che misurano Shapley, LIME e permutation). Su MASTER dji L = 1 la `net` di IG coincide con Shapley esatto (scarti ≤ 0,01), la `importance` è più sbilanciata (position 28%, candle 7%). Quale usare nello Step 7 si motiva nello Step 6 (C1) |
| 2026-11-04 | Dati spostati da `code/data/data/<universo>` a `code/data/<universo>` (dove li cerca `config.DATA_ROOT`); `.venv` ricreato (torch 2.11.0+cu128) |
