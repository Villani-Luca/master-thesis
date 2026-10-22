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
- [ ] **B1. Misura del costo** di IG (50 passi, entrambi i target) su MASTER e FactorVAE, dji y2020:
  secondi per data, memoria GPU. Stima del batch sulle 117 run e sulle run seed 0/5 future.
  *Fatto quando:* c'è un numero in ore, scritto nel registro.
- [ ] **B2. `xai/base.py`**: protocollo `Explainer`, `explain(adapter, batch, target) -> attr[N,T,F]`
  con `target` in {`pred`, `topk_margin`}; date di ribilanciamento di L = 5 (le stesse per ogni
  orizzonte); baseline zero; salvataggio float16 nel results store.
- [ ] **B3. Metodi** in `xai/`: IG e Saliency (captum), permutation importance per famiglia
  (mescola la famiglia tra i titoli dello stesso giorno, misura ΔRankIC e cambio del top-k),
  attention. Prima tutto su MASTER × dji × y2020, poi gli altri cinque modelli. Casi delicati: grafo
  di HIST, FactorVAE, attention da far restituire. **Al massimo un giorno ciascuno**: se non basta,
  quel modello resta senza attention (annotarlo nel registro).
- [ ] **B4. SHAP/LIME** a gruppi (famiglie) su dji y2020, tutti i modelli.
- [ ] **B5. `xai/aggregate.py`**: |a| normalizzato per campione, importanza per feature, lag,
  famiglia; segno medio; dispersione; ranking → `global.parquet` per run e metodo.
- [ ] **B6. `scripts/explain.py`** (resumable come `train_rolling.py`) e batch sulle 117 run.
- [ ] **B7. Notebook `05_xai_single_model.ipynb`** con osservazioni.
- *Fatto quando:* `global.parquet` per IG e permutation su tutte le 117 run, per entrambi i target.

### C. Step 6–9 sulle attribuzioni
- [ ] **C1. Step 6 ridotto**: sanity check a pesi randomizzati, accordo tra metodi (IG, permutation,
  SHAP, LIME su dji y2020), deletion curve → notebook 06 con la scelta motivata dei metodi.
- [ ] **C2. Step 7 completo**: `analysis/agreement.py` (Spearman, Kendall, Jaccard top-k, RBO),
  `analysis/variance.py` (fattori: modello, anno, universo; seed per MASTER subito, per tutti dopo il
  rientro; orizzonte se arrivano in tempo le attribuzioni L = 1, 20), confronto `pred` vs
  `topk_margin`, diagnosi dati vs modello scritta, notebook 07.
- [ ] **C3. Step 8 leggero**: per anno, 2020 e 2022 contro gli anni calmi; IC, rendimenti, quote
  delle famiglie, deriva anno su anno; notebook 08.
- [ ] **C4. Step 9.1**: fragilità delle decisioni (margini dello Step 4 contro il rumore tra seed:
  subito su MASTER, poi su tutti).

### D. Preparazione al rientro della griglia
- [ ] **D1. Notebook 03 e 04** generici su 5 universi × 3 orizzonti: rieseguibili senza modifiche.
- [ ] **D2. Procedura di rientro**: estrarre il pacchetto, `grid.py status` = 1242/1242,
  `export_results.py`, `run_portfolio.py`, notebook 03/04, `compare_finbench.py`, poi
  `explain.py` sulle nuove run L = 5 (seed 0/5) e, se c'è tempo, L = 1 e 20 su dji, nasdaq100, sx5e.
- [ ] **D3. Pacchetti parziali**: chiedere all'amico un `pack_results.sh` ogni ~3 giorni, per far
  partire lo XAI sulle nuove run senza aspettare la fine.

### E. App (Step 11, ridotta)
- [ ] **E1. `3_Explain`**: ranking globale + famiglie, heatmap T × F per data e titolo, attention.
- [ ] **E2. `4_Compare`**: ranking affiancati, heatmap di accordo, dispersione tra seed.

### F. Scrittura (in inglese, nelle pause dei batch)
- [ ] **F1. Metodologia**: dati e universi, modelli, protocollo rolling, regole di selezione, portafoglio.
- [ ] **F2. Prestazioni e portafogli** (notebook 03/04; numeri aggiornati al rientro).
- [ ] **F3. Metodi XAI e loro valutazione.**
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
