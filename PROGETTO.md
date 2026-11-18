# Progetto della tesi: stato, piano, decisioni, riesecuzioni e guida

Questo è l'unico file di lavoro del progetto. Il 2026-11-18 vi sono stati uniti `PIANO.md` (piano, decisioni, registro),
`RIESECUZIONE.md` (comandi e riesecuzioni) e `THESIS_GUIDE.md` (guida tecnica, in inglese), poi cancellati: le versioni
precedenti sono nella storia di git (`git log -- PIANO.md`). Il contenuto è stato riordinato, non riassunto; le parti
superate sono nell'appendice «Storico».

Per evitare sigle doppie, le voci che erano in `RIESECUZIONE.md` hanno nuovi prefissi: **R1–R5** (invariate),
**S1–S4** (universi grandi sulla H200, prima E1–E4), **Z1–Z3** (chiusura, prima D1–D3), **RA/RB/RC** (rientro della
griglia e primo batch XAI, prima A/B/C). Le sigle **A–F** sono quelle del piano di lavoro di ottobre, **D1–D6** le
decisioni del 22/10, **Step 0–12** i passi della guida tecnica.

## Indice

1. Come lavorare su questo file
2. Stato al 18/11/2026 e cose da fare
3. Obiettivo, vincoli e calendario
4. Decisioni (D1–D6) e registro delle decisioni
5. Riesecuzioni: XAI su tutte le date (R1–R5)
6. XAI sugli universi grandi, sulla H200 (S1–S4)
7. Chiusura (Z1–Z3)
8. Piano di lavoro di ottobre–novembre (A–F)
9. Guida tecnica (Step 0–12, in inglese)
10. Appendice: storico

## 1. Come lavorare su questo file

Per Claude (o chi riprende il lavoro):

1. Leggi prima la sezione 2 (stato e cose da fare), poi le parti della guida tecnica che servono (Step 5–9, 11) e
   `docs/model_notes.md` per i dettagli dei modelli. La struttura dei risultati XAI è in `docs/xai_struttura.md`.
2. Spunta le voci (`- [x]`) con una riga su cosa è stato fatto e quando. Le decisioni nuove vanno nel registro
   (sezione 4), con la data; lo stato della sezione 2 si aggiorna quando cambia.
3. Regole:
   - non modificare modelli, iperparametri, seed, protocollo o regole di selezione: lo XAI spiega i
     checkpoint così come sono;
   - non cancellare e non sovrascrivere nulla in `code/results/Regression/` (`--force` mai);
   - lo XAI scrive solo nelle sue cartelle (§2.2 della guida tecnica);
   - le figure per la tesi vanno in `latex-thesis/images/results/` con nomi fissi;
   - la tesi è **in inglese**, in prosa rifinita (Luca la rivede e la riscrive a mano); codice e commenti nello
     stile del repo; questo file è in italiano, la guida tecnica resta in inglese.
4. Se una scelta non è coperta da questo file e cambia i risultati, chiedi prima.

Tutti i comandi vanno lanciati dalla radice del repository, con l'ambiente attivo (`.venv\Scripts\activate`).
Avanzamento XAI: `python code/scripts/grid.py status --xai`. Errori: `grep -l Traceback logs/xai/task*.log` e
`grep Traceback logs/*.err`. Su Windows FinFormer esce con 0xC0000409 a fine processo: è innocuo se il run risulta
spiegato.

## 2. Stato al 18/11/2026 e cose da fare

| Cosa | Stato |
|---|---|
| Training della griglia | **1242/1242** run (5 universi × L = 1, 5, 20 × 3 seed × anni di test): 1125 sulla H200, 117 riaddestrati sulla RTX 5090 (rientro il 4/11, riaddestramento finito il 5/11) |
| Portafogli top-k | fatti per tutta la griglia, con la regola sulle aperture inaffidabili (RA2) |
| XAI sul core (DJI, NASDAQ-100, EURO STOXX 50) | **756/756** run, tutte le date di ribilanciamento, 6 metodi (IG, Saliency, permutation, Shapley e LIME sulle famiglie, pesi interni), entrambi i target; finito il 10/11 alle 08:38 (R1) |
| Valutazione dei metodi | 54/54 run del core (R1c), notebook 06, cap. 5 §5.7 |
| XAI su S&P 500 e STOXX Europe 600 (H200) | **0/486**: nessun pacchetto ricevuto al 13/11; non si sa se il server ha iniziato (S2–S4) |
| Notebook | 03–09 rieseguiti sui risultati finali del core; 07 e 08 anche con il fattore orizzonte (R5) |
| Tesi | capitoli 1–9 in bozza, numeri del core aggiornati alle spiegazioni su tutte le date |
| App | `Home`, `1_Data`, `2_Models`, `6_Portfolio`; mancano `3_Explain` e `4_Compare` (E1, E2) |

Cose da fare, in ordine:

- [ ] **S2–S4.** Sulla H200: verificare che lo XAI degli universi grandi sia partito (S2), ricevere i pacchetti (S3),
  poi l'analisi su tutti e cinque gli universi e i numeri dei cap. 6 e 9 (S4). Da qui non si vede lo stato del server.
- [ ] **Revisione a mano** del testo nuovo: cap. 5 §5.7 "Evaluating the Explanations" (R3) e il paragrafo "The horizon
  moves the explanation as much as the architecture" del cap. 6 §6.3 (R5).
- [ ] **TODO residui della tesi** (`grep -n "TODO" latex-thesis/chapters/*.tex`): nel cap. 5 la quota del gradiente
  che passa dagli altri titoli (tutti e sei i modelli), LIME contro Shapley per titolo, convergenza di IG, ore GPU e
  verifica sulla H200 (`check_xai.py`); cap. 9 e introduzione sullo scope (dopo S3); contributo 5 dell'introduzione
  (dopo l'app); esempio di spiegazione locale nel cap. 6 (facoltativo).
- [ ] **E1, E2.** Pagine dell'app `3_Explain` e `4_Compare`, poi il cap. 8 (F5).
- [ ] **Z1, Z2.** Copia dei risultati sul drive e commit.
- Calendario: bozza completa al relatore verso metà gennaio, scadenze Esse3 il 19/01 e 28/01, laurea il 09/02/2027
  (sezione 3).

## 3. Obiettivo, vincoli e calendario

- **Scadenza (aggiornata il 2026-11-05): seduta di laurea del 09/02/2027.** Domanda su Esse3 dall'11/12/2026
  all'11/01/2027; altre due scadenze Esse3 il 19/01/2027 e il 28/01/2027 (da confermare quale sia il caricamento
  del file: per analogia con la sessione di dicembre, 12 giorni prima della seduta, dovrebbe essere il 28/01).
  La prenotazione per la seduta del 01/12/2026 va annullata su Esse3 (i 32 euro non si ripagano; vedi il
  promemoria su ingmo.unimore.it). La vecchia scadenza del 18–19/11/2026 non vale più.
- Il relatore chiede una tesi consistente (4–6 mesi di lavoro) e **lo XAI su tutti e 5 gli universi e su tutti e
  3 gli orizzonti**; vuole vedere i risultati a pochi giorni di distanza, come per la griglia.
- Domanda di ricerca: come le feature guidano le predizioni e l'allocazione top-k, e se
  l'instabilità viene dai **dati** o dal **design dei modelli** (Step 7: è il risultato centrale e
  non si taglia).
- Tagliati per la scadenza del 18/11 (sviluppi futuri nella tesi): Step 9.2, Step 10, pagina app `5_Over_time`,
  riferimento LightGBM/TreeSHAP. Con la laurea a febbraio si possono riconsiderare: decidere con il relatore dopo
  i primi risultati sugli universi grandi. SHAP/LIME non è più tagliato: dal 5/11 su tutti i run.
- Orizzonti L = 1, 5, 20 **obbligatori** per tutti e 5 gli universi (li produce la griglia H200).

### Calendario in vigore

Calendario rivisto il 2026-11-05 sera (laurea il 09/02/2027, XAI su 5 universi × 3 orizzonti):

| Date | Lavoro |
|---|---|
| 5–7 nov | fine XAI L = 1/20 sui 3 universi core (`xai3_*`); notebook 06; pacchetto per la H200 (universi grandi) e avvio sul server |
| ~8–12 nov | **primi risultati al relatore**: core completo (L = 1, 5, 20) e primi universi grandi; tempi misurati dell'XAI su S&P 500 e STOXX Europe 600 |
| nov – dic | XAI su tutte le date: core sul PC (`grid.py run --xai`, L = 5 ~2–3 giorni, poi L = 1/20 ~1–2 settimane), universi grandi sulla H200 (sezione S, risultati ogni 2–3 giorni); notebook 07 e 08 estesi ai fattori universo e orizzonte; capitoli 5–7 e 9 aggiornati; app e cap. 8 |
| 11/12 – 11/01 | domanda di laurea su Esse3 |
| **~metà gennaio** | **bozza completa al relatore**, prima delle scadenze Esse3 |
| 19/01 e 28/01 | scadenze Esse3 (probabilmente il file della tesi il 28/01: da confermare) |
| **09/02/2027** | **seduta di laurea** |

Stima dei tempi (da misurare sul server, passo 6 di START_HERE_XAI.md; la H200 da sola è più veloce della RTX 5090 condivisa): universi grandi a L = 5 una data su due 2–6 settimane,
L = 1 e 20 una data su quattro altrettanto, quindi 4–12 settimane in tutto. Se le misure danno il caso
peggiore, ripiego: una data su quattro anche a L = 5 sugli universi grandi (da decidere con le misure).

## 4. Decisioni

### Decisioni del 22/10 (D1–D6)

| # | Domanda | Decisione |
|---|---|---|
| D1 | Run seed 0/5 dello stage 4 | perse. XAI subito sulle 117 run; il confronto tra seed (Step 7) usa MASTER finché non arrivano i seed 0/5 dalla H200, poi si estende |
| D2 | Dove gira lo XAI | macchina con CUDA (sviluppo e batch) |
| D3 | Metodi | Integrated Gradients (+ Saliency) su tutte le run; permutation importance cross-sectional; attention dove il modello la espone facilmente; SHAP/LIME solo su dji y2020, tutti i modelli, come controllo di IG |
| D4 | Cosa si spiega | sia il punteggio (`target="pred"`) sia l'appartenenza al top-k (`target="topk_margin"`) |
| D5 | Granularità di permutation e SHAP/LIME | le 8 **famiglie** di `xaifin.data.features.FAMILIES` (IG resta a risoluzione piena T × F e si aggrega dopo) |
| D6 | Bozza al relatore | verso il 10/11 |

### Registro delle decisioni

In ordine di inserimento (non strettamente cronologico).

| Data | Decisione |
|---|---|
| 2026-10-22 | Scadenza 18/11, bozza ~10/11; scope tagliato (anche nella guida tecnica) |
| 2026-11-05 | **Shapley e LIME su tutti i 1242 run** (tutte le date, entrambi i target), decisione di Luca; prima solo dji 2020 come controllo di IG. `explain.py`: tutti e 6 i metodi di default; `grid.py --xai` conta un run come fatto solo con anche `shapley/` e `lime/`. Costo misurato su DJI (s per giorno, i due metodi e i due target): FactorVAE 45, DiscoverPLF 13, gli altri 2–3; cresce in modo lineare con i titoli (una passata per coalizione su tutto il giorno), meno di IG: sul core circa +50% di tempo. Pacchetto H200 ricostruito (riferimento di `check_xai.py` con Shapley e LIME). Il processo separato Shapley/LIME dji 2020 (`xai_c2`) è stato fermato dopo 1 run perché ridondante con lo scheduler. I primi 5 task dello scheduler locale, partiti prima del cambio (DiscoverPLF L = 5), non fanno Shapley/LIME: a fine giro lo scheduler va rilanciato una volta e completa solo quelli. **I dati non sono più sul server**: si manda `E:\Thesis\data.zip` (38 GB, identico ai dati locali per i file che servono; lo zip dei dati ricostruito qui è stato fermato e cancellato) |
| 2026-11-05 | **XAI su tutte le date di ribilanciamento** (~52 l'anno, ogni orizzonte), decisione di Luca: con la laurea a febbraio cade il motivo del campionamento (una data su due a L = 5, una su quattro a L = 1/20, scelto il 4–5/11 solo per i tempi). `REBALANCE_STRIDE = 1` (`xai/base.py`), default di `explain.py` e di `grid.py --xai`, anche nel pacchetto per la H200 (ricostruito, riferimento di `check_xai.py` rifatto con 53 date). Le uscite a date campionate (7,1 GB, core L = 5 completo, L = 1/20 parziale, Shapley/LIME dji 2020) sono spostate in `code/results/xai_sampled_dates/` (non cancellate): servono a mostrare quanto cambia campionando. Prima prova (MASTER dji L = 5 seed 42 2020, 53 contro 27 date): Spearman feature 0,997 (9/10 in comune nel top 10), famiglie 1,0, lag 0,95. Core rilanciato alle 18:26 con `grid.py run --xai --universes dji nasdaq100 sx5e --workers 5` (log `logs/xai_core.out`, `logs/xai/task<N>.log`; prima L = 5, poi L = 1/20) e Shapley/LIME dji 2020 (`logs/xai_c2.out`). Da rifare dopo: notebook 06, 07, 08 e i numeri del cap. 6 e 9 (cambiano poco). Spazio: ~40 GB di tensori sul PC, ~230 GB sul server (guida aggiornata: ~350 GB liberi) |
| 2026-11-05 | **Laurea spostata alla seduta del 09/02/2027** (accordo con il relatore: tesi consistente, 4–6 mesi di lavoro). **XAI su tutti e 5 gli universi e su tutti e 3 gli orizzonti** (richiesta del relatore). Stessa densità del core: L = 5 una data di ribilanciamento su due, L = 1 e 20 una su quattro. Lanciati alle 17:55 `xai4_a` (MASTER, MATCC; sp500 e sxxp, L = 5; 54 run) e `xai4_b` (FactorVAE, HIST, FinFormer, DiscoverPLF; 108 run); L = 1 e 20 sugli universi grandi dopo la fine di `xai3_*` (sezione S). Calendario rivisto sopra |
| 2026-11-05 | **Universi grandi spiegati sulla H200** (server del training); questo PC fa i 3 universi core. `xai4_*` fermati alle 18:00 senza output. Pacchetto: `make_bundle.py` esteso (`--runs` con i checkpoint, `--universes`, `--xai-reference`, `--guide`, `--no-code`); `grid.py --xai` (stessi task del training con `explain.py` al posto di `train_rolling.py`; run fatto = tutti gli output XAI presenti; stride 2 a L = 5 e 4 a L = 1/20; `--ram-per-task`); `xai.sh`, `pack_xai.sh` (leggero senza tensori / completo; log in `logs/xai_server/`), `check_xai.py` (rifà sul server MASTER dji 2020 e lo confronta con quello di questo PC), guida `START_HERE_XAI.md`, `requirements-xai.txt` (+ captum). Gli output delle due macchine si uniscono senza conflitti (universi diversi) |
| 2026-11-05 | **Step 7, correzione dei numeri dell'accordo per feature** (notebook 07 e 08 rieseguiti, diagnosi e Observations scritte). Due errori di misura, corretti: (a) `variance.pair_agreement` confrontava anche modelli di gruppi diversi a livello di feature, sui 3 nomi in comune (OPEN0, HIGH0, LOW0): ora `_feature_level` esclude le coppie che condividono meno di metà dei nomi; (b) le 63 feature di mercato (42 in Europa), a zero per costruzione con la baseline media, gonfiavano lo Spearman di MASTER e MATCC con zeri appaiati: ora escluse anche a livello di feature (come già la famiglia `market`). Numeri corretti, feature: seed 0,59, universo 0,60, anno 0,66, **modello 0,27** (prima 0,67/0,62/0,71/0,19); Jaccard top 10 tra modelli 0,10. Famiglie dentro Alpha158 (7 famiglie): modello 0,74, universo 0,79, seed 0,82, anno 0,84; in Alpha360 sempre 1 (3 famiglie, ordine dato dal numero di colonne: 296 momentum, 60 volume, 4 candle). **Per modello, feature, seed vs altri modelli del gruppo**: MASTER 0,29 vs 0,28, FactorVAE 0,27 vs 0,22, FinFormer 0,33 vs 0,09 (il seed conta quanto l'architettura); MATCC 0,84, HIST 0,95, DiscoverPLF 0,84 vs 0,28–0,37. **Stesso seed, anni consecutivi**: MASTER 0,72, FactorVAE 0,56 (contro 0,29 e 0,27 tra seed nello stesso anno): l'inizializzazione segna la spiegazione più della finestra di training. Drift tra anni senza mercato: feature 0,71 (calmo-calmo) vs 0,70 (con anno volatile). Scomposizione della varianza invariata (usava già le quote nette senza mercato) |
| 2026-11-05 | **Step 7 (notebook 07 e 08, L = 5, 252 run, definitivi)**. (1) **Alpha360: le feature dominanti sono un artefatto dell'ordine delle colonne.** Le più importanti sono quelle che il layout mette negli ultimi passi della sequenza ricorrente (HIST: 52% dell'importanza negli ultimi 5 passi su 60, 0,9% nei primi 5; FinFormer 32%; DiscoverPLF 23%; uniforme 8%): DiscoverPLF ha in testa VWAP9, 19, 29, 39, 49, 59, tutte nel passo 59. (2) Accordo per fattore, feature: seed 0,67, anno 0,71, universo 0,62, **modello 0,19**; famiglie: tutte 0,90–0,94 (solo 3 famiglie in comune tra gruppi: misura grossolana). (3) Scomposizione della varianza delle quote per famiglia: Alpha158 **seed 43%**, interazioni 31%, modello 11%, anno 9%, universo 6%; Alpha360 modello 49%, seed 27%; tutti i modelli sulle famiglie comuni: modello 92% (ma è la composizione dei due set di input: momentum 18% vs 70%). Nessuna feature Alpha158 nel top 10 di metà delle run. (4) `pred` vs `topk_margin` identici (1,0); ΔRankIC vs cambio del top-k −0,12…0,13. (5) Regimi: quote delle famiglie quasi uguali tra anni volatili e calmi (scarti ≤ 3 punti); l'accordo tra anni consecutivi non cala nelle transizioni con un anno volatile (famiglie 0,93 vs 0,92, feature 0,74 vs 0,75); FinFormer è il meno stabile tra anni sulle feature (0,34) |
| 2026-11-05 | **XAI esteso a L = 1 e L = 20 sui 3 universi core**, tutti i modelli e seed, una data di ribilanciamento su quattro (~13 giorni l'anno; `explain.py --horizons 1 20 --stride 4`, processi `logs/xai3_a|b|c.out`), per il fattore orizzonte dello Step 7. Gli universi grandi (S&P 500, STOXX Europe 600) restano senza spiegazioni: ~5× il costo del core per orizzonte, tutto su tutto ~3–4 settimane. Messaggio al relatore per confermare lo scopo |
| 2026-11-05 | La scadenza formale del file della tesi è il 19/11; si tiene il 18/11 come obiettivo di lavoro. Calendario rivisto (sezione Calendario) |
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
| 2026-11-05 | **Riaddestramento finito** (11:16, 1242/1242). Riproduzione: 99/117 bit-identiche; le 18 diverse sono tutte e sole le MATCC (non determinismo, vedi sotto): RankIC ±0,008 in media (max 0,024), MSE ±0,016, un cambio di segno su 18, RankIC medio 0,011 → 0,009. Si usano le run nuove. Batch XAI dei seed 0/5 finito (135 run), `--reaggregate` e pesi interni fatti |
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

## 5. Riesecuzioni: XAI su tutte le date (R1–R5)

Le spiegazioni ora coprono **tutte** le date di ribilanciamento (registro delle decisioni, 5/11). Le uscite a date campionate
sono in `code/results/xai_sampled_dates/`; tutto ciò che le usava va rifatto quando il core è rispiegato:

- [x] **R1. Core su questo PC** (completo il 10/11 alle 08:38, 756/756): `grid.py run --xai --universes dji nasdaq100 sx5e --workers 5` (lanciato alle 18:26,
  `logs/xai_core.out`; avanzamento: `python code/scripts/grid.py status --xai --universes dji nasdaq100 sx5e`). Prima
  L = 5 (252 run), poi L = 1 e 20 (504). Su Windows FinFormer esce con 0xC0000409 a fine processo: il task risulta FAILED
  nel log ma le uscite sono complete (lo conferma `status --xai`); rilanciare lo stesso comando rifà solo ciò che manca.
- [ ] **R1b. Processo extra** `xai_fast` (5/11, 19:00; `logs/xai_fast.out`): MASTER e MATCC, core, L = 1, 5 e 20,
  tutti i metodi (252 run), presi dalla coda dello scheduler, che li salterà. Dal 5/11 `grid.py` legge
  `logs/xai/WORKERS` (numero di processi modificabile mentre gira); vale dal prossimo rilancio dello scheduler.
  5/11 19:12: `xai_extra1` (`logs/xai_extra1.out`): HIST, core, L = 1 e 20 (84 run), dalla coda dello scheduler.
  5/11 19:27: `xai_extra2` (`logs/xai_extra2.out`): FactorVAE, core, L = 1 e 20 (84 run), dalla coda dello scheduler.
  5/11 21:25: **cache delle coalizioni** in `xai/shap_lime.py`: ogni coalizione si valuta una volta per run, giorno e
  target, e LIME (200 estrazioni fra 128 o 8 coalizioni possibili) riusa quelle di Shapley. Valori identici (differenza
  0 su MASTER e HIST), Shapley+LIME 2–2,4 volte più veloci. I tre processi extra sono stati riavviati con il codice nuovo
  (`logs/xai_fast2.out`, `xai_extra1b.out`, `xai_extra2b.out`; saltano ciò che era già scritto). Pacchetto H200 ricostruito.
  5/11 23:12: `xai_fast2` fermato (memoria GPU 31,9 GB stabile; era l'extra con più memoria, 3,4 GB su NASDAQ-100).
  MASTER e MATCC: DJI completo per MASTER su tutti gli orizzonti, il resto da riprendere con lo stesso comando
  quando la memoria torna sotto 26 GB (i 3 DiscoverPLF NASDAQ-100 dello scheduler tengono 22,5 GB).
  6/11 10:22: `xai_extra1b` (HIST L = 1/20) fermato per memoria GPU 31,7 GB stabile (era l'extra con più memoria, 1,0 GB,
  appena passato a NASDAQ-100). Fatti: HIST DJI L = 1 e L = 20 completi (30 run). Da riprendere con lo stesso comando
  (`explain.py --models HIST --universes dji nasdaq100 sx5e --horizons 1 20`) quando la memoria torna sotto 26 GB.
  6/11 12:11: ripresa di MASTER e MATCC come `xai_fast3` (PID 27576, `logs/xai_fast3.out`; memoria GPU 19,3 GB,
  GPU all'82%). HIST L = 1/20 resta da riprendere (`logs/xai_extra1c.out`).
  6/11 13:42: valutazione estesa (`xai_eval`, PID 20148 e lanciatore 38564) fermata per memoria GPU 31,56 GB stabile
  (11 letture in un minuto; era l'extra con più memoria, 5,0 GB). Fatti 7/54 (DJI L = 5 per 6 modelli, DiscoverPLF
  NASDAQ-100 L = 5). Memoria scesa a 27,0 GB. Da riprendere con lo stesso comando quando la memoria torna sotto 26 GB
  (salta i run con summary.json).
  6/11 16:57: `xai_fast3` (PID 38452 e lanciatore 27576) fermato. La memoria GPU letta era 31,44 GB, poco sotto la
  soglia, ma era satura: 5,3 GB già spostati nella memoria condivisa (RAM di sistema), di cui 4,1 GB del solo
  `xai_fast3` (MASTER NASDAQ-100 L = 20, 7,0 GB dedicati), e tutti i run andavano 1,7–2x più lenti (FinFormer
  EURO STOXX 50 2830 s invece di 1420, FinFormer DJI 3191 invece di 1840). Dopo lo stop: 25,0 GB, 0,55 GB condivisi.
  Fatti: MASTER completo su DJI e NASDAQ-100 L = 1/5, più NASDAQ-100 L = 20 seed 0 2020. Da riprendere con lo stesso
  comando (`logs/xai_fast3.out`). Da qui conta anche la memoria condivisa ('\GPU Adapter Memory(*)\Shared Usage'
  sopra ~1 GB = GPU satura) oltre alla soglia di 31,5 GB.
  6/11 19:26: MASTER e MATCC ripresi come `xai_fast4` (`logs/xai_fast4.out`; memoria GPU 18,9 GB dopo la fine di
  FinFormer NASDAQ-100 L = 5 seed 0 e 5).
  7/11 04:43: HIST L = 1/20 ripreso come `xai_extra1c` (`logs/xai_extra1c.out`; memoria GPU 13,1 GB dopo la fine
  della valutazione estesa).
  7/11 15:13: MASTER e MATCC completi (252/252, `xai_fast4` chiuso alle 14:57). Lanciato FinFormer L = 1/20 come
  `xai_extra3` (`logs/xai_extra3.out`, PID 16388; memoria GPU 7,1 GB, GPU all'82%, 7 processi).
  7/11 21:12: `xai_extra1c` (HIST L = 1/20) fermato: memoria GPU 31,85 GB stabile e 3,7 GB di memoria condivisa,
  perché i task DiscoverPLF NASDAQ-100 L = 1 dello scheduler, passati al 2021, sono saliti a 7 GB ciascuno. Era
  l'extra con più memoria (1,1 GB). Fatti: HIST NASDAQ-100 L = 1 seed 0 e seed 42 2020–2023 (9 run); il run seed 42
  2024 era a metà e senza file scritti. Dopo: 31,5 GB e 2,9 GB condivisi. Da riprendere con lo stesso comando
  (`explain.py --models HIST --universes dji nasdaq100 sx5e --horizons 1 20`, `logs/xai_extra1d.out`) quando la
  memoria torna sotto 26 GB.
  7/11 21:42: `xai_extra3` (FinFormer L = 1/20) fermato: RAM libera 4,1–4,9 GB (sotto 5) e memoria GPU 31,74 GB
  stabile. I task DiscoverPLF NASDAQ-100 dello scheduler, passati al 2021, tengono 7,7 GB di RAM e fino a 7 GB di GPU
  ciascuno. FinFormer era l'extra con più memoria GPU (0,43 GB contro 0,31 di FactorVAE; 2,6 GB di RAM). Fatti:
  FinFormer DJI L = 1 seed 0 e 42 completi, seed 5 2020–2021 (12 run); il run seed 5 2022 era a metà. Dopo: RAM 7,1 GB,
  memoria GPU 31,5 GB. Da riprendere con lo stesso comando (`logs/xai_extra3b.out`) quando RAM e memoria lo permettono.
  7/11 22:27: anche `xai_extra2b` (FactorVAE L = 1/20, l'ultimo extra) fermato: RAM libera 1,3 GB e memoria GPU 31,67 GB
  stabile. Tutti e 5 i task DiscoverPLF NASDAQ-100 dello scheduler sono ora sui run del 2021 e tengono ~9 GB di RAM
  ciascuno (~45 GB su 63,6). Fatti: FactorVAE NASDAQ-100 L = 1 seed 0 completo e seed 42 2020–2021 (37/84 del
  processo); il run seed 42 2022 era appena iniziato. Dopo: RAM 3,5–4,3 GB (commit 79 su 94 GB, quindi c'è ancora
  margine nel file di paging), memoria GPU 31,3 GB. Restano solo i 5 task dello scheduler, che non si toccano. Da
  riprendere con lo stesso comando (`logs/xai_extra2c.out`) per primo, quando la RAM libera torna sopra 10 GB.
  8/11 17:38–18:07: lo scheduler ha finito DiscoverPLF NASDAQ-100 L = 1 (15/15) e L = 20 seed 0 e 5; ~5,1 h per run
  sotto saturazione. Ora è su DiscoverPLF DJI L = 1/20 (~47 min per run) e NASDAQ-100 L = 20 seed 42.
  8/11 18:40: FactorVAE L = 1/20 ripreso come `xai_extra2c` (PID 26284; memoria GPU 10,6 GB, RAM libera 37,7 GB, GPU 83%).
  8/11 22:35: lo scheduler ha finito DiscoverPLF DJI L = 1 (15/15, ~56 min per run) e DJI L = 20 seed 0; ora è su DJI
  L = 20 seed 5 e 42, EURO STOXX 50 L = 1 seed 0 e 5, NASDAQ-100 L = 20 seed 42.
  9/11 00:39: HIST L = 1/20 ripreso come `xai_extra1d` (PID 22968; memoria GPU 17,1 GB, RAM libera 33 GB, GPU 82%).
  9/11 09:50: lo scheduler ha finito DiscoverPLF L = 1/20 su tutto il core (DiscoverPLF completo) e lanciato FinFormer
  NASDAQ-100 L = 1/20 (task 234–236, 240–241; ~6 GB di GPU e ~7,7 GB di RAM ciascuno). FinFormer extra non lanciato
  (gruppo in coda allo scheduler). HIST NASDAQ-100 L = 1 completo.
  9/11 12:40: `xai_extra2c` (FactorVAE) fermato di nuovo: RAM libera 2,5 GB, memoria GPU 31,2 GB e 3,1 GB condivisi.
  Era l'extra con più memoria GPU (2,05 GB; 4,9 GB di RAM). Fatti: FactorVAE NASDAQ-100 L = 1 completo, L = 20 seed 0
  2020–2022 (il 2023 ha IG e Saliency scritti). Dopo: RAM 9,3 GB, memoria GPU 30,4 GB. Da riprendere con lo stesso
  comando (`logs/xai_extra2d.out`) quando RAM e memoria lo permettono.
- [x] **R1c. Valutazione dei metodi estesa** (5/11 sera): `code/scripts/evaluate.py` ripete sanity check e deletion
  curve (IG e Saliency, 'own' e 'joint') su un run per modello × universo × orizzonte (seed 42, 2022, 5 date): 54 run
  sul core, ~1–15 min l'uno; uscite in `code/results/xai/evaluation/`. Provato su MASTER dji L = 5 (sanity 0,90,
  AOPC own IG 0,89 contro 0,001 a caso). Da lanciare quando c'è margine di GPU (`logs/xai_eval.out`); sugli universi
  grandi lo fa il server dopo la griglia. **Lanciata il 6/11 alle 11:58** (PID 38564, staccata; GPU all'82% e 18 GB
  di memoria GPU dopo la fine dei task DiscoverPLF NASDAQ-100 L = 5). Fermata alle 13:42 per memoria GPU (vedi R1b),
  a 7/54; da riprendere. **Ripresa il 6/11 alle 17:56** (PID 25440, `logs/xai_eval2.out`; memoria GPU 25,6 GB).
  **Finita il 7/11 alle 04:28: 54/54.** Mediane (sanity = Spearman delle attribuzioni addestrato vs pesi casuali per
  feature, alto = la spiegazione dipende dal modello; AOPC = calo dell'uscita togliendo gli input per importanza):

  | Modello | sanity | AOPC IG own | a caso | AOPC Saliency own | a caso | AOPC IG joint | a caso |
  |---|---|---|---|---|---|---|---|
  | DiscoverPLF | 0,44 | 0,313 | 0,003 | −0,015 | 0,003 | −0,003 | 0,000 |
  | FactorVAE | 0,42 | 0,244 | −0,003 | −0,009 | −0,003 | 0,241 | −0,002 |
  | FinFormer | 0,37 | 0,482 | −0,014 | −0,124 | −0,014 | 0,474 | −0,059 |
  | HIST | 0,98 | 0,039 | 0,001 | 0,001 | 0,001 | 0,009 | 0,001 |
  | MASTER | 0,87 | 0,894 | 0,001 | −0,014 | 0,001 | 0,692 | 0,002 |
  | MATCC | 0,71 | 1,825 | −0,011 | −0,060 | −0,011 | 0,642 | 0,000 |

  Per universo (sanity / AOPC IG own): DJI 0,59 / 0,48, NASDAQ-100 0,60 / 0,28, EURO STOXX 50 0,46 / 0,49; per
  orizzonte: L = 1 0,59 / 0,22, L = 5 0,47 / 0,54, L = 20 0,52 / 0,52. Sanity da 0,23 (FinFormer DJI L = 1) a 0,98.
  Esito: **IG own batte il caso in 54/54 run** (scarto mediano +0,41); IG joint in 49/54 (non su DiscoverPLF, dove il
  percorso congiunto non sposta l'uscita); **Saliency no: sopra il caso solo in 12/54, scarto mediano −0,009**, cioè il
  gradiente locale non ordina gli input per effetto sull'uscita (già visto nella prova su MASTER DJI, −0,007). HIST
  supera sempre il sanity (0,97–0,98) ma con AOPC piccolo (0,02–0,07, tranne EURO STOXX 50 L = 5 con 0,58): la sua
  uscita cambia poco togliendo input singoli. Casi con AOPC a caso lontano da 0: DiscoverPLF NASDAQ-100 L = 20
  (−0,107) e FinFormer NASDAQ-100 L = 20 (+0,104), IG comunque molto sopra. Da riportare nel cap. 5/6 (sanity e
  faithfulness su tutto il core al posto della sola prova; Saliency come metodo di confronto, non affidabile per
  l'ordinamento).
- [x] **R1d. Shapley/LIME dei 24 DiscoverPLF L = 5 parziali** (6/11 16:30): DJI seed 0 e 5 (+ seed 42 2020) e
  NASDAQ-100 (15) hanno IG, Saliency e permutation, non Shapley/LIME (fatti dal codice vecchio dello scheduler).
  Costano poco (~18 s per metodo e target, `params_*.json`; explain.py salta i metodi già scritti), ma lo scheduler
  vecchio non li rifà e il suo rilancio arriva solo a fine griglia (~10/11). Da lanciare appena la memoria GPU
  scende sotto 26 GB, come primo candidato: `explain.py --models DiscoverPLF --universes dji nasdaq100 --horizons 5`
  (`logs/xai_plf_fill.out`): con questo il core L = 5 è completo intorno al 7/11 pomeriggio invece che il 10/11.
  **Lanciato il 6/11 alle 17:11** (PID 10560; memoria GPU 25,0 GB, GPU all'81%). **Finito alle 17:42**: 29 run
  completati (uno era già completo), ~60 s l'uno su DJI e ~90 s su NASDAQ-100. DiscoverPLF L = 5 completo su tutti
  e tre gli universi core.
- [x] **R2. Shapley e LIME**: dal 5/11 sera sono tra i metodi di default, quindi li fa lo scheduler su ogni run
  (il processo separato `xai_c2` è stato fermato). **Quando lo scheduler finisce, rilanciarlo una volta** (stesso
  comando di R1): i primi 5 task, partiti prima del cambio (DiscoverPLF L = 5), completano solo Shapley e LIME.
- [x] **R3. Quando L = 5 è finito**: notebook 05, 06, 07, 08 rieseguiti; Observations e diagnosi ricontrollate;
  `thesis_figures.py res_inputs res_agreement res_agreement_models res_variance`; numeri dei cap. 6 e 9 aggiornati.
  7/11 18:41: lanciati in background notebook 07, 08, 05, 06 (`nbconvert --execute --inplace`) e poi le tabelle res_*
  (`logs/r3_run.ps1`, uscita in `logs/r3.out`). 07, 08 e 05 rieseguiti senza errori alle 18:43.
  7/11 19:27: cap. 6 aggiornato dai notebook 05, 07, 08 (differenze piccole). Cambiano: permutation, la classifica
  delle famiglie per calo del RankIC ora è debolmente legata a quella per impatto sul portafoglio (Spearman medio
  0,01–0,36 per modello, prima −0,12–0,13; il testo dice "at most weakly related"), calo massimo 0,011, 12 coppie su 30
  in cui il RankIC sale; varianza Alpha158 architettura 10% (prima 11%), interazioni 32%; accordo per modello e regimi
  ritoccati di un centesimo. Restano: tabelle res_* (dopo il 06) e i numeri del testo che ne derivano (§6.2, famiglie
  0,74–0,84 di §6.3), i TODO del notebook 06 (cap. 5 e §6.3, §6.6).
  7/11 20:24: notebook 06 e tabelle res_* finiti (`ALL DONE` in `logs/r3.out`). 20:28: testo aggiornato.
  - Tabelle: differenze di un punto (price-volume 9–13%, ultimo giorno 8–11% in §6.2; MATCC/HIST/DiscoverPLF).
  - Cap. 5, §5.7 "Evaluating the Explanations": scritti i risultati al posto del TODO (sanity e deletion curve dai 54
    run di R1c; accordo fra metodi e impostazioni dal notebook 06, DJI 2020 seed 42) e le scelte (IG joint, quota
    netta). Aggiornati anche baseline zero (0,89 MASTER, 0,53–0,98), percorso own/joint (0,999–1,000 per MASTER,
    MATCC, FactorVAE, HIST; **0,90 DiscoverPLF e 0,82 FinFormer**, prima "0,986–1,000") e quota netta vs Shapley.
  - Cap. 6: pesi vicini all'inizializzazione (mediane 4% MASTER, 5% FactorVAE, 7% HIST) e sanity (HIST 0,98, MASTER
    0,87, MATCC 0,71, gli altri 0,37–0,44) dai 54 run; tolti i due TODO del notebook 06.
  - Cap. 9: 0,26 e 10% (RQ2).
  Da guardare a mano: il nuovo §5.7 (testo nuovo, non solo numeri) e le due novità del notebook 06 che il testo prima
  non diceva: MATCC si allontana dai pesi iniziali (98%) ma le attribuzioni cambiano poco; per DiscoverPLF il percorso
  congiunto si discosta da quello per titolo e IG joint non batte il caso in 4 run su 9. Restano i TODO di cap. 5 non
  legati al notebook 06 (quota del gradiente dagli altri titoli, LIME vs Shapley per titolo, convergenza di IG, ore GPU).
- [x] **R4. Verifica del campionamento** (una frase nel cap. 5): classifiche con tutte le date contro una su due / una su
  quattro, per ogni run del core (`xai/` contro `xai_sampled_dates/`).
  7/11 18:50: `python code/scripts/check_date_sampling.py` (tabella per run in `code/results/analysis/date_sampling.csv`).
  IG su 329 run (L = 5 tutti, L = 1 e 20 quelli già fatti): Spearman delle feature mediana 0,996 (min 0,94), stessa
  classifica delle famiglie nel 95% dei run (Spearman famiglie ≥ 0,93); Saliency 0,999 (min 0,96), 97%. Shapley e LIME
  solo su 19 run in comune (la prima versione li aveva su pochi run). Frase nel cap. 5 (sezione "The days") aggiornata;
  da rieseguire a R5, quando L = 1 e 20 sono completi.
  7/11 18:35: L = 5 completo sul core (252/252).
- [x] **R5. Quando L = 1 e 20 sono finiti**: notebook 07 e 08 con il fattore orizzonte (S4 per gli universi grandi).
  10/11 08:38: core completo, 756/756 (scheduler vecchio: "162 ok, 0 failed"). 12:38: rilanciato una volta con il
  codice nuovo (`logs/xai_core2.out`): 0 task da fare, quindi niente buchi (anche Shapley/LIME dei DiscoverPLF L = 5).
  R4 rieseguito sul core completo: IG su 342 run, Spearman feature mediana 0,995 (min 0,93), stessa classifica delle
  famiglie nel 95%; Saliency 0,999 (min 0,96), 98%. Frase del cap. 5 aggiornata.
  R5: `xaifin/analysis/variance.py` ha il fattore `horizon` (coppie di run che differiscono solo nell'orizzonte;
  orizzonte come quinto fattore della scomposizione, solo se la tabella ne ha più d'uno: a L = 5 i numeri non cambiano,
  verificato). Notebook 07 (sezione 7) e 08 (sezione 5) con le celle nuove, rieseguiti. Risultati (core, IG, quota netta):
  famiglie Alpha158 tra orizzonti 0,69 (come tra modelli, 0,70; seed 0,82), feature 0,59 (come tra seed); 1 vs 5 0,71,
  5 vs 20 0,81, 1 vs 20 0,54; scomposizione Alpha158 su 378 run: seed 34%, interazioni 39%, orizzonte 12% (47% per il
  candle, che in MASTER e MATCC scende dal 13% a L = 1 al 6–8% a L = 20), modello 6%; Alpha360 orizzonte 5%.
  Regimi: a ogni orizzonte nessuna famiglia si sposta di più di 4 punti, anni consecutivi 0,90–0,94.
  **Testo nuovo da rivedere a mano**: cap. 6, paragrafo "The horizon moves the explanation as much as the architecture"
  (§6.3) e una frase in §6.5.
  11/11 06:30: Diagnosis del notebook 07 e Observations del notebook 08 allineate alle spiegazioni su tutte le date
  (erano rimasti i numeri della versione a date campionate: 0,26, Jaccard 0,09, 32% / 10%, permutation "debolmente
  legata", spostamenti dei regimi) e completate con l'orizzonte (07 punto 7, 08 punto 5). Lavoro locale finito:
  restano solo i punti del server (S2–S4).

Le voci RC1, RC2, RC4, RC4b, RC5 dello storico restano spuntate per la versione a date campionate; R1–R5 le sostituiscono.

## 6. XAI sugli universi grandi, sulla H200 (S1–S4)

Decisione del 2026-11-05 (registro delle decisioni): XAI su 5 universi × 3 orizzonti. I tre universi core su questo PC
(processi `xai3_*`), i due grandi sulla H200 del server del training (486 run). I processi `xai4_*` lanciati qui alle
17:55 sono stati fermati alle 18:00 prima di finire un run: nessun file scritto.

- [x] **S1. Pacchetto** (5/11): `C:\Users\villa\Downloads\xaifin_xai_bundle.zip` (1,4 GB: codice, i 486 checkpoint,
  il run di riferimento MASTER dji 2020 per `check_xai.py`, guida `START_HERE.md` = `code/scripts/remote/START_HERE_XAI.md`)
  e, per i dati, `E:\Thesis\data.zip` (38 GB, del 19/10: tutti e 5 gli universi; confrontato il 5/11 con `code/data`,
  i 40 file che servono al server sono identici per nome, dimensione e checksum). Il server non ha più i dati del training.
  Ricostruirli con i due comandi nel docstring di `code/scripts/remote/make_bundle.py`.
- [ ] **S2. Sul server**: Claude segue `START_HERE.md` (ambiente, `check_xai.py`, prova e misura, lancio di
  `xai.sh`, cioè `grid.py run --xai --universes sp500 sxxp`, tutte le date). Farsi dare le misure del passo 6 (secondi per run, RAM)
  e aggiornare la stima nel calendario.
- [ ] **S3. Ricezione** ogni 2–3 giorni del pacchetto leggero (`xaifin_xai_light_*.tar.gz`, senza tensori delle
  attribuzioni), alla fine di quello completo: estrarre dalla radice del repo (`tar -xzf xaifin_xai_light_<...>.tar.gz`).
  Scrive solo in `code/results/xai/*/*/{sp500,sxxp}/` e `logs/xai_server/`. Controllo:
  `python code/scripts/grid.py status --xai`; leggere `logs/xai_server/xai_check.txt`.
- [ ] **S4. Analisi**: notebook 07 e 08 con tutti gli universi e gli orizzonti (fattori universo e orizzonte nella
  scomposizione della varianza), poi capitoli 6 e 9 e le tabelle `res_*` di `thesis_figures.py` (ora fissate a L = 5
  e universi core).

## 7. Chiusura (Z1–Z3)

- [ ] **Z1. Copia sul drive** di `code/results/Regression/`, `code/results/portfolio/` e `code/results/xai/`
  (i file grandi sono fuori da git: checkpoint, pickle, predizioni, holdings, attribuzioni).
- [ ] **Z2. Commit** (i file piccoli: metriche, config, storie del training, global XAI, tabelle, notebook, tesi).
- [x] **Z3.** Aggiornare le voci di PIANO.md (D2, B6, B7, C1, C2) e cancellare questo file, o archiviarlo.
  → 18/11: PIANO.md, RIESECUZIONE.md e THESIS_GUIDE.md uniti in questo file e cancellati; voci del piano aggiornate.

## 8. Piano di lavoro di ottobre–novembre (A–F)

In ordine. Ogni voce ha il criterio di "fatto".

### A. Avvio sulla macchina CUDA
- [x] **A1. Codice e dati**: repo aggiornato (commit/push dal PC Windows, pull qui), `pip install -e code`,
  `grid.py status` mostra le 117 run (altrimenti trasferire i checkpoint).
- [x] **A2. Ambiente**: `00_environment_check.ipynb` gira senza errori (CUDA, captum, shap, lime).

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
- [x] **B4. SHAP/LIME** a gruppi (famiglie) su dji y2020, tutti i modelli.
  → 18/11: superato, Shapley e LIME su tutti i run (registro, 5/11; R2).
  → 2026-11-04: `xai/shap_lime.py`: Shapley **esatto** sulle famiglie (2^7 = 128 coalizioni per giorno,
  famiglia assente = titolo medio del giorno per tutti i titoli; efficienza esatta, 1e-8) e LIME sulla stessa
  rappresentazione; `explain.py --methods shapley lime`. Prova (MASTER dji L = 1): ~1,7 s/giorno, LIME vs
  Shapley 0,88–0,92 per titolo; quote nette di IG per famiglia ≈ Shapley (scarti ≤ 0,01). Da lanciare su
  dji y2020 seed 42, tutti i modelli, a riaddestramento finito.
- [x] **B5. `xai/aggregate.py`**: |a| normalizzato per campione, importanza per feature, lag,
  famiglia; segno medio; dispersione; ranking → `global.parquet` per run e metodo.
  → 2026-11-04: anche per fonte e orizzonte; `global_<target>.parquet` per run e metodo.
- [x] **B6. `scripts/explain.py`** (resumable come `train_rolling.py`) e batch sulle 117 run.
  → 2026-11-04: script fatto. Batch partito sulle run L = 5 con checkpoint (seed 0/5 dalla H200) in 3
  processi staccati (`logs/xai_a|b|c.out`); il seed 42 e i MASTER seguono a training finito.
- [x] **B7. Notebook `05_xai_single_model.ipynb`** con osservazioni.
  → 18/11: rieseguito su tutte le date e con osservazioni (RC3, R3).
  → 2026-11-04: notebook scritto (6 sezioni), collaudato su MASTER dji L = 1 seed 5 y2022; da eseguire su
  MASTER dji L = 5 seed 42 y2020 a riaddestramento finito e scrivere le osservazioni. Prime indicazioni
  (run di prova): IG su `pred` e `topk_margin` quasi identici (Spearman 0,9998); permutation: il ranking
  delle famiglie per cambio del top-k concorda con IG (0,93), quello per ΔRankIC no (0,43–0,57); own vs
  joint 0,998, baseline zero vs media 0,84, completezza mediana 0,5%; gate di MASTER vs IG 0,21.
- *Fatto quando:* `global.parquet` per IG e permutation su tutte le 117 run, per entrambi i target.
  → 18/11: B6, C1–C4 e D2–D3 chiusi dal lavoro di novembre: XAI su tutti i 756 run del core e su tutte le date (R1),
  valutazione estesa (R1c), notebook 05–09 rieseguiti (R3, RA4b), orizzonte (R5); D3 non serve più (griglia rientrata il 4/11).

### C. Step 6–9 sulle attribuzioni
- [x] **C1. Step 6 ridotto**: sanity check a pesi randomizzati, accordo tra metodi (IG, permutation,
  SHAP, LIME su dji y2020), deletion curve → notebook 06 con la scelta motivata dei metodi.
  → 2026-11-04: `xai/evaluation.py` (sanity check contro i pesi iniziali, deletion curve "joint" e "own",
  AOPC) scritto e provato su MASTER, HIST, FactorVAE (prime evidenze nel registro). Notebook
  `06_xai_evaluation.ipynb` scritto e collaudato (dji seed 5 y2022, 5 modelli; MASTER manca finché non è
  riaddestrato): vedi registro. Mancano l'esecuzione sul seed 42 e la scelta motivata.
- [x] **C2. Step 7 completo**: `analysis/agreement.py` (Spearman, Kendall, Jaccard top-k, RBO),
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
- [x] **C3. Step 8 leggero**: per anno, 2020 e 2022 contro gli anni calmi; IC, rendimenti, quote
  delle famiglie, deriva anno su anno; notebook 08.
  → 2026-11-04: notebook `08_periods_and_regimes.ipynb` scritto e collaudato (regimi del cap. 3: 2020 e 2022
  alta volatilità, 2021/2023/2024 calmi): RankIC per anno e regime contro la volatilità dell'indice,
  rendimento in eccesso sull'equal weight per anno, quote nette delle famiglie per anno e differenza tra
  regimi, deriva anno su anno delle spiegazioni (anni consecutivi, transizioni calme vs con anno volatile).
  Da rieseguire a batch XAI completo e dopo `--reaggregate`; mancano le osservazioni.
- [x] **C4. Step 9.1**: fragilità delle decisioni (margini dello Step 4 contro il rumore tra seed:
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
- [x] **D2. Procedura di rientro**: estrarre il pacchetto, `grid.py status` = 1242/1242,
  `export_results.py`, `run_portfolio.py`, notebook 03/04, `compare_finbench.py`, poi
  `explain.py` sulle nuove run L = 5 (seed 0/5) e, se c'è tempo, L = 1 e 20 su dji, nasdaq100, sx5e.
  → 2026-11-04: pacchetto (`xaifin_results_luca-gpu-priv-0_20261103_0809`) copiato, 1242/1242, nessun
  traceback nei log (in `logs/grid_h200/`); `export_results.py` fatto; portafogli fatti per le run con pickle.
  **In corso:** riaddestramento sulla RTX 5090 delle 117 run pre-griglia (vedi registro), poi
  `run_portfolio.py` (anche `--k 5`, `--force` sui 18 seed 42 senza parquet), notebook 03/04, `compare_finbench.py`.
  Processi staccati: `logs/grid_retrain_117.out`, `logs/portfolio_fix.out` (ricalcolo dopo la regola sulle
  aperture, vedi registro); finiti `logs/portfolio_k5.out`, `logs/portfolio_sxxp.out`.
  **Da fare, in ordine** (versione operativa con i comandi e i momenti: sezioni «Riesecuzioni» e «Storico», RA–RC):
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
- [x] **D3. Pacchetti parziali**: chiedere all'amico un `pack_results.sh` ogni ~3 giorni, per far
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
- [x] **F2. Prestazioni e portafogli** (notebook 03/04; numeri aggiornati al rientro).
  → 2026-11-05: cap. 6 §6.1 (prestazioni su 1242 run, figure `res_rankic`, `res_horizons`) e scheletro delle sezioni
  6.2–6.6 con i contenuti attesi; cap. 7 risultati con `pf_performance`, `pf_fragility`, `pf_equity`,
  `pf_fragility_scatter`; tutto da `code/scripts/thesis_figures.py`.
- [ ] **F3. Metodi XAI e loro valutazione.**
  → 2026-11-04: prima bozza di `05_xai_methodology.tex` (~2900 parole: cosa si spiega, metodi, baseline,
  percorso joint/own, aggregazione lorda/netta, costi, piano di valutazione). Numeri da run di prova marcati
  `% TODO`; manca la sezione con i risultati dello Step 6. Allineate due frasi del cap. 2 (baseline, percorso).
  Non compilata: LaTeX non è installato su questo PC.
- [x] **F4. Risultati XAI, diagnosi, conclusioni** (dopo C2).
  → 18/11: cap. 6 e 9 scritti e aggiornati alle spiegazioni su tutte le date (R3, R5).
- [ ] **F5. Introduzione** (alla fine); il cap. 3 segnaposto va sostituito.
  → 2026-11-05: introduzione aggiornata (numeri finali, griglia core/estesa, contributi su fragilità e
  riproducibilità); restano le risposte anticipate alle RQ. Cap. 9: limiti e sviluppi futuri scritti; mancano
  le risposte alle domande di ricerca (dopo la diagnosi dello Step 7).

## 9. Guida tecnica (Step 0–12, in inglese)

The technical plan of the thesis, from `THESIS_GUIDE.md`. Its section numbers (§0–§4) and steps are kept as they were, so the references in the code ("Step 5.1", "§2.2") still apply; the deadline box and the calendar of §3 are in the appendix (superseded).

This is the working plan for turning [`docs/papers/Outline Tesi.pdf`](docs/papers/Outline%20Tesi.pdf) into code, notebooks, a reusable framework and the final graphical application. Each step says **why** it exists (which part of the outline it serves), **what to build** (notebook, package module, app page), **what it produces** (files), and **when it counts as done**.

> **Research question.** Stock-prediction models have weak predictive power (accuracy ≈ 50%, IC ≈ 0.0x). That instability carries over into long-only top-k portfolios. *How do input features drive predictions, how does that turn into allocation decisions, and does the instability come from the **data** or from the **model design**?*

### 0. How to use this guide

- Work through the steps in order. Every step has three layers:
  1. **Notebook** (`code/notebooks/NN_*.ipynb`): where you explore, explain and produce the thesis figures. Notebooks stay thin: they import logic, run it and plot the results.
  2. **Package** (`code/xaifin/`): the reusable framework. Logic moves here once it works in the notebook, and the notebook then imports it back.
  3. **Application** (`code/app/`): a Streamlit page per step that reads the same package and the same results store. The app contains **no logic of its own**.
- Tick the checkboxes (`- [x]`) as you go. The tracker in §4 gives the overall status.
- Each step lists the **thesis artefacts** (figures and tables) it should produce, so the writing in `latex-thesis/` can happen alongside the code.

#### Mapping outline → steps

| Outline item | Guide step(s) |
|---|---|
| *Prima parte* 1–3: read the survey, the model papers and the XAI literature | Step 0.2 |
| **Step 1**: comparable model groups | Step 1 |
| **Step 2**: train with FinBench → checkpoints | Steps 2, 3 |
| **Step 2**: SHAP/LIME/time-series XAI → rank and compare features | Steps 5, 6, 7 |
| **Step 3**: qualitative analysis, data vs model | Step 7 |
| *Prima parte* 4: framework that applies XAI and saves results and metrics | Steps 2, 5, 6 |
| *Prima parte* 5: rolling-window tests, volatile periods | Steps 3, 8 |
| Allocation (long-only top-k) | Steps 4, 9 |
| **Step 4 (optional)**: counterfactual stability of the portfolio | Step 9 |
| *Seconda parte*: data-modification sensitivity tests | Step 10 |
| **Conclusion**: GUI to load models, show attributions and compare architectures | Step 11 (built incrementally from Step 1) |

---

### 1. The pipeline at a glance

```
OHLCV  ──►  Alpha158 / Alpha360 features  ──►  robust z-score (fit on train)  ──►  model (MASTER, HIST, …)
                                                                                     │
                     ┌──────────────────── predictions ŷ[date, ticker] ◄──────────────┘
                     │
      ┌──────────────┼──────────────────────────┬──────────────────────────────┐
      ▼              ▼                          ▼                              ▼
 metrics (IC,   long-only top-k            attributions A[date,ticker,T,F]   counterfactuals
 RankIC, MSE)   portfolio (CAGR, Sharpe,   (SHAP, IG, LIME, attention, …)    (how much must x or ŷ
                turnover)                   │                                 change to flip top-k?)
                                            ▼
                         rank features ─► compare across models / seeds / years / universes
                                            ▼
                               diagnosis: data-limited signal vs model design
```

---

### 2. Target architecture and conventions

#### 2.1 Repository layout

The skeleton exists (2026-09-29): every folder and `xaifin` subpackage below is in place, and each `__init__.py` lists its planned modules. The modules themselves are created step by step.

```
master-thesis/
├── README.md                   # overview + setup
├── PROGETTO.md                 # this file: plan, status, decisions, re-runs, this guide
├── docs/
│   ├── papers/                 # the papers + Outline Tesi.pdf
│   ├── model_groups.md         # FinBench models grouped by input data
│   └── model_notes.md          # per-model notes + FinBench issues
├── latex-thesis/               # the thesis
└── code/                       # everything below
```

```
code/
├── pyproject.toml              # makes `xaifin` installable: pip install -e code
├── requirements.txt
├── data/                       # local datasets (not in git)
├── finbench/                   # FinBench clone: read-only reference (not in git)
├── tests/                      # pytest
├── xaifin/                     # the framework ("Prima parte" item 4)
│   ├── config.py               # paths, universes, model groups, rolling windows, seeds, k
│   ├── data/
│   │   ├── loading.py          # read alpha158/360, market, constituents, labels
│   │   ├── normalization.py    # RobustZScore (fit on train only), cross-sectional z-score
│   │   ├── datasets.py         # daily cross-sectional datasets -> DayBatch
│   │   ├── features.py         # feature names + feature FAMILY mapping (Alpha158/360/market)
│   │   ├── quality.py          # samples with unusable prices, discarded when config.CLEAN_DATA
│   │   └── summary.py          # Step 1 data summaries, cached in results/data_summary/
│   ├── models/
│   │   ├── base.py             # ModelAdapter protocol (see 2.3)
│   │   ├── master.py  matcc.py  factorvae.py        # Alpha158 group
│   │   ├── hist.py  discoverplf.py  finformer.py     # Alpha360 group
│   │   ├── lgbm_reference.py   # optional tree baseline with exact TreeSHAP
│   │   └── registry.py         # "MASTER" -> adapter class
│   ├── training/
│   │   ├── trainer.py          # generic loop, early stopping on valid RankIC
│   │   ├── metrics.py          # IC, RankIC, ICIR, MSE, MAE, R2 (scipy-free fallbacks)
│   │   └── rolling.py          # rolling-window orchestration
│   ├── portfolio/
│   │   ├── topk.py             # long-only top-k holdings from predictions
│   │   ├── backtest.py         # returns, CAGR, Sharpe, MaxDD, turnover
│   │   └── counterfactual.py   # Step 9
│   ├── xai/
│   │   ├── base.py             # Explainer protocol + masking/baseline utilities
│   │   ├── gradient.py         # Saliency, IntegratedGradients, GradientSHAP, DeepLIFT (Captum)
│   │   ├── shap_lime.py        # KernelSHAP / PermutationSHAP / LIME on feature GROUPS
│   │   ├── permutation.py      # cross-sectional permutation importance on IC / top-k
│   │   ├── temporal.py         # time-step / lag-window occlusion, TimeSHAP-style
│   │   ├── intrinsic.py        # attention weights (MASTER/MATCC/FinFormer), HIST concepts
│   │   ├── aggregate.py        # per-feature / per-lag / per-family global importance
│   │   └── evaluation.py       # faithfulness, stability, sanity checks (Step 6)
│   ├── analysis/
│   │   ├── agreement.py        # Spearman, Kendall, top-k Jaccard, RBO
│   │   ├── variance.py         # seed vs model vs year vs universe decomposition
│   │   └── regimes.py          # volatility regimes from market index data
│   ├── perturb/                # Step 10: OHLCV noise, ablations, shifts
│   ├── store.py                # read/write the results store (2.2); single source of truth
│   └── viz.py                  # plotly figures shared by notebooks and app
├── scripts/                    # CLI entry points for batch runs (grid over models/years/seeds)
│   ├── train_rolling.py
│   ├── run_portfolio.py
│   ├── explain.py
│   └── analyze.py
├── notebooks/                  # one notebook per step (see each step)
├── app/                        # Streamlit GUI (Step 11)
│   ├── Home.py
│   └── pages/
└── results/                    # results store (2.2)
```

#### 2.2 Results store layout (one layout, read by notebooks, scripts and the app)

```
code/results/
├── Regression/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<YEAR>/
│   ├── model.pth                  # best-valid checkpoint
│   ├── config.json                # full run config (dates, hparams, feature list, git hash)
│   ├── metrics.json               # test metrics
│   ├── train_history.csv
│   └── results_sl<T>_pl<L>.pkl    # FinBench-compatible: preds, labels, pred_date, last_date, tickers
├── portfolio/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/top<k>/
│   ├── holdings.parquet           # date, ticker, weight, score, rank
│   ├── returns.parquet
│   └── metrics.json               # CAGR, Sharpe, MaxDD, Vol, turnover
├── xai/<METHOD>/<MODEL>/<universe>/sl<T>_pl<L>/seed<S>/y<YEAR>/
│   ├── attributions.npz           # attr[D,N,T,F] float16 on REBALANCE dates + dates, tickers, feature_names
│   ├── global.parquet             # aggregated importance per feature / lag / family
│   ├── params.json                # method, baseline, n_samples, target (pred or top-k margin)
│   └── xai_metrics.json           # faithfulness / stability scores (Step 6)
└── analysis/…                     # cross-model tables, figures exported for the thesis
```

> ✅ **Migrated 2026-09-29.** The first MASTER runs (seed 42, y2020, from the old `master_model.ipynb`) followed this layout for dji, nasdaq100 and sp500. They have no `config.json`. The sp500 run had only `model.pth`: its test evaluation never completed. **Moved to `results/legacy/` on 2026-10-06** (other selection rule, no data-quality filter); Step 3 retrains them.

`.gitignore` excludes `code/results/xai/**/attributions.npz` (large files). `global.parquet` and the metrics stay under version control.

#### 2.3 The key abstraction: `ModelAdapter`

Every model, whatever its architecture, is wrapped so that the XAI, portfolio and app code sees the same interface:

```python
@dataclass
class DayBatch:
    date: pd.Timestamp            # prediction ("last input") date
    tickers: list[str]            # N stocks in the cross-section that day
    x: torch.Tensor               # [N, T, F] normalized features (model's own feature space)
    y: torch.Tensor               # [N] label (forward L-day return, as the model was trained)
    extras: dict                  # fixed non-feature inputs: market_value, stock2concept, …

class ModelAdapter(Protocol):
    name: str                     # "MASTER"
    group: str                    # "alpha158" | "alpha360"
    feature_names: list[str]      # length F (for MASTER/MATCC: 157 alpha + market gate features: 63 US, 42 EU)
    seq_len: int                  # T
    def build(self, cfg) -> None
    def load(self, run_dir: Path) -> None
    def day_batches(self, split: str) -> Iterator[DayBatch]
    def forward(self, x: torch.Tensor, extras: dict) -> torch.Tensor   # [N,T,F] -> [N], differentiable, eval mode
```

As implemented (`xaifin/models/base.py`): the adapter is built from a `RunConfig` (model, universe, test year, seed, T, L, `clean`, hyper-parameter overrides), which replaces `build(cfg)`. It also carries how FinBench trains the model, so the Step 3 trainer stays generic: `training_loss(batch)`, `configure_optimizer()`, `target(batch)` (the labels the predictions are scored against) and the loop settings in `hparams` (`n_epochs`, `grad_clip`, `scheduler_step`). `DayBatch.y` is the label as the model's training loss receives it (raw return for MASTER/MATCC/HIST/DiscoverPLF, daily z-score for FactorVAE and FinFormer). Each adapter implements `model_forward`; the base `forward` wraps it and turns cuDNN off when gradients are on, because cuDNN's GRU can't backpropagate in eval mode (predictions then differ by up to ~1e-4 from the cuDNN ones). Alpha360 adapters take `x = [N, 1, 360]` in the CSV's column order and `extras`: `market_value` and `stock2concept` (HIST, DiscoverPLF), `adjacency` (FinFormer).

Points to keep in mind:
- **Cross-sectional models.** MASTER, MATCC, FactorVAE, HIST and FinFormer mix information *across stocks* on the same day, so stock *i*'s prediction depends on the other stocks' inputs. Always explain a **whole day** (`x` = `[N,T,F]`) and attribute stock *i*'s output to *its own* inputs by default, holding the others fixed. Cross-stock attribution (how much stock *j* influences *i*) is an optional extra.
- **Non-feature inputs** (`extras`): HIST and DiscoverPLF also consume `<universe>_inc_matrix.npz` (stock→concept) and `<universe>_market_cap.csv`. Hold these fixed during XAI and document it.
- **Market gate features** (MASTER/MATCC, from `<nation>_market.csv`) are identical for all stocks on a date. Treat them as their own `market` family.
- **Baseline for masking.** After the robust z-score, `0` means "training median", which makes it a natural and defensible baseline. Also test a per-day cross-sectional mean baseline (Step 6).

#### 2.4 Conventions

- `code/finbench/` is a **read-only reference** (it is gitignored). Copy code out of it, never import from it or edit it. This is the convention already used in [`master_model.ipynb`](code/notebooks/master_model.ipynb).
- All paths come from `xaifin.config`. Nothing is hard-coded in notebooks.
- **Data-quality filter** (decided 2026-09-29): samples whose input or label reads a ticker of the wrong company, a stale price, a price below 0.01 or a one-day jump beyond ×3 are discarded in training, validation and test (`xaifin/data/quality.py`, `config.CLEAN_DATA = True`). It removes 0.07–0.9% of the samples (2.3% for DJI, all of them `PRG.US`). FinBench keeps them, so the filter is turned off only to reproduce FinBench in the Step 2 equivalence test.
- Every run writes `config.json` with seed, dates, hyper-parameters and `git rev-parse HEAD`.
- Seeds: `0, 5, 42`, the same three as the FinBench paper (§4.1). Three seeds are needed to separate seed noise from model differences in Step 7.
- Default horizon: **T=20, L=5** (FinBench's "weekly" configuration). Alpha360 models use an effective lookback of **T=1**: one 360-vector per day, which already covers 60 days × 6 series. Optional extras: (T=5, L=1) and (T=60, L=20).
- Universes (as the outline suggests): **DJI, NASDAQ100, SX5E**. SP500/SXXP are optional scale-up.

#### 2.5 Rolling windows (FinBench protocol)

| Test year | Train | Valid | Test |
|---|---|---|---|
| y2020 | 2015–2018 | 2019 | 2020 |
| y2021 | 2016–2019 | 2020 | 2021 |
| y2022 | 2017–2020 | 2021 | 2022 |
| y2023 | 2018–2021 | 2022 | 2023 |
| y2024 | 2019–2022 | 2023 | 2024 |

(The data runs into Jan 2026, so a y2025 window is possible as an extension.)

---

### Step 0: Environment and literature

#### 0.1 Environment (blocker for everything that uses SHAP)

**Current state (checked 2026-09-29):** in `.venv` (Python 3.12), `torch 2.11 + cu128` works with CUDA. The scipy "Application Control policy" block is resolved: `scipy 1.18`, `scikit-learn`, `shap 0.52`, `captum 0.9`, `lime`, `streamlit 1.64` and `pyarrow 25` all import, and a KernelSHAP / Integrated Gradients smoke test passes. `xaifin` is installed in editable mode.

- [x] Fix the scipy block. Options, from least to most invasive:
  - reinstall scipy (`pip install --force-reinstall scipy`); a different wheel is sometimes not flagged;
  - use a fresh env from conda-forge (differently built binaries);
  - run the project in **WSL2** or on the RTX-5090 GPU machine mentioned in `requirements.txt`;
  - turn off Smart App Control. ⚠️ On Windows 11 it cannot be turned back on without resetting Windows, so decide this deliberately.
- [x] Add to `code/requirements.txt`: `shap`, `captum`, `lime`, `streamlit`, `pyarrow`, and optionally `timeshap`.
- [x] Create `code/pyproject.toml` and run `pip install -e code` so that `import xaifin` works from notebooks, scripts and the app.
- [x] **Notebook `00_environment_check.ipynb`**: imports, CUDA, versions, data files, and `captum`/`shap`/`lime` smoke tests on a tiny MLP (23 checks, all passing).

**Done when:** `00_environment_check.ipynb` runs top to bottom with no errors.

#### 0.2 Reading (*Prima parte* 1–3)

The PDFs in `docs/papers/`:

| File | Paper | What to extract |
|---|---|---|
| `1-s2.0-S095741741630029X-main.pdf` | CI & Financial Markets survey (ESWA 2016) | the classic pipeline: data → features → model → trading |
| `13208-AAAI24.LiT.pdf` | MASTER (AAAI 2024) | task, label, market gate, T/S attention |
| `3627673.3679715.pdf` | MATCC (CIKM 2024) | differences vs MASTER |
| `12027-DuanY.pdf` | FactorVAE (AAAI 2022) | prior/posterior factors, which output to explain |
| `2110.13716v2.pdf` | HIST | concept modules, stock2concept matrix |
| `Discovering_Predictable_Latent_Factors_….pdf` | DiscoverPLF (TKDE 2023) | latent factors, extra losses |
| `Finformer_….pdf` | FinFormer (BigData 2023) | static-dynamic graph, CCC loss |
| `NIPS-2017-a-unified-approach-….pdf` | SHAP | Shapley axioms, KernelSHAP, DeepSHAP |
| `2939672.2939778.pdf` | LIME (KDD 2016) | local surrogate, sampling |
| `Explainable_AI_in_Portfolio_Management_….pdf` | XAI in PM review | where XAI is used in allocation |
| `1-s2.0-S2405844024161269-main.pdf` | Explainable DL for stock trend | a worked finance XAI example |
| *(missing)* | Qlib (arXiv 2009.11189) | origin of Alpha158/360. The reference implementation is in `code/finbench/Evaluation/features/alpha158.py` and `alpha360.py` |

- [x] For each model, write a half-page note: **what is predicted** (label = forward L-day return on adj close), **inputs**, **normalizations** (see the FinBench README table), **loss**, **whether it is cross-sectional**, and **which internal signals could be inspected** (attention, factors, concepts). → [`docs/model_notes.md`](docs/model_notes.md), which also lists **6 FinBench implementation issues** to resolve before Step 2.
- [x] XAI literature search, focused on **time-series** and, if any exist, **stock-trend** specific methods. Candidates to check: Integrated Gradients, DeepLIFT/DeepSHAP, TimeSHAP, Temporal Saliency Rescaling, Dynamask, FIT, WinIT, the "attention is (not) explanation" debate, Rashomon-set / explanation-disagreement papers, and counterfactual methods (Wachter et al., DiCE). Save the notes to `latex-thesis/chapters/02_second_chapter.tex` (related work) and to `bibliography.bib`. → Chapter 2 drafted ("Background and Related Work"); 46 references, all verified.

**Thesis artefacts:** related-work chapter; one table of the six models (task, inputs, normalization, loss, cross-sectional yes/no).

---

### Step 1: Comparable model groups and universes (Outline Step 1)

**Why:** explanations can only be compared between models that see the same data under the same protocol.

**Already done:** [`docs/model_groups.md`](docs/model_groups.md) groups all FinBench models into five groups. The core of the thesis is the two alpha groups, which are exactly the regression models listed in the outline:

| Group | Feature space | Models | Input per stock |
|---|---|---|---|
| **A: Alpha158** | 157 engineered factors (Qlib's 158 minus `VWAP0`; FinBench has no VWAP input) + market gate features for MASTER/MATCC (63 US, 42 EU) | MASTER, MATCC, FactorVAE | `[T=20, F=157(+63 US / +42 EU)]` |
| **B: Alpha360** | 6 raw normalized series (CLOSE, OPEN, HIGH, LOW, VOLUME, VWAP) × 60 lags | HIST, DiscoverPLF, FinFormer | `[60, 6]` |
| *(opt.) Reference* | Alpha158, tabular (last day) | LightGBM | `[F=157]`, with exact TreeSHAP |

Within a group, compare features one to one. Across groups, compare at the **family** level (Step 5.4).

#### Tasks
- [x] `xaifin/config.py`: `UNIVERSES`, `NATION` map (dji/nasdaq100→us, sx5e→eu), `MODEL_GROUPS`, `ROLLING_WINDOWS`, `SEEDS`, `TOP_K` (e.g. 10, and 5 for DJI's ~30 stocks), `DATA_ROOT`, `RESULTS_ROOT`. → Also `CORE_UNIVERSES`, `SL_PL_CONFIGS`, `rolling_window(year)`.
- [x] `xaifin/data/features.py`: feature names plus a **family mapping**. → Done as `catalog(universe)`: family (8), source (5) and horizon (3) per feature; names checked against the CSV headers of all 5 universes. Alpha360 overlaps Alpha158 only on momentum, volume and candle, so compare the groups by source and horizon.
  - Alpha158: `kbar` (KMID, KLEN, KUP, KLOW, KSFT…), `price` (OPEN0, HIGH0, LOW0), and each rolling operator (ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP/N/D, SUMP/N/D, VMA, VSTD, WVMA, VSUMP/N/D) × window {5,10,20,30,60}.
  - Higher-level **semantic families**, shared across groups: `trend/momentum`, `volatility/range`, `volume`, `price-volume correlation`, `market`. Also horizon buckets `short (≤5d)`, `medium (10–20d)`, `long (30–60d)`.
  - Alpha360: series × lag bucket (lag 0–4 short, 5–19 medium, 20–59 long).
- [x] **Notebook `01_universes_and_data.ipynb`**: for each universe, show tickers over time (constituents), date coverage, NaN rates per feature, label distribution per year, correlation clusters among the Alpha158 features (many are near-duplicates, which matters for SHAP and for counterfactuals), and index realized volatility per year (input for Step 8). → Done for all five universes, with the thesis tables of groups and universes and a data-quality section (extreme labels, tickers mapped to the wrong company). Summaries are cached by `xaifin.data.summary` in `results/data_summary/`, charts are in `xaifin/viz.py`. Key findings in the notebook's *Observations*: the EU membership history starts on 2020-09-30 (so there is no 2020 test year for SX5E/SXXP); DJI's "Procter & Gamble" (`PRG.US`) is PROG Holdings in every test year; wrong tickers and bad prices in SXXP and SP500, now discarded by the data-quality filter (§2.4); 20 exactly redundant Alpha158 features (SUMP/N/D, VSUMP/N/D).
- [x] **App page `1_Data.py`**: pick a universe → coverage chart, feature table with family, correlation heatmap, volatility timeline. → Tabs: Coverage, Features, Labels, Redundancy, Market volatility, each with its table. Uncached universes get a "Compute summary" button.

**Done when:** the config and family mapping exist, the notebook renders every universe, and the app page works.
**Thesis artefacts:** table of groups; table of universes (N stocks, period, noise/volatility); feature-correlation figure.

---

### Step 2: Unified framework: data and model adapters

**Why:** training, XAI and the GUI all need to load any model and run it on any day through one interface (§2.3).

#### Tasks
- [x] Move the MASTER code from `master_model.ipynb` into the package. The notebook was deleted from the working tree; read it with `git show 7fcec03:code/notebooks/master_model.ipynb`. Targets: `data/loading.py` (constituent filter, market merge, `extract_labels`; the market path is already done), `data/normalization.py` (`RobustZScoreNormalization`), `data/datasets.py` (the daily dataset that yields `DayBatch`; it drops the samples of `quality.discarded_samples` after building the input sequences when `config.CLEAN_DATA`), `models/master.py`, and `training/metrics.py` (the scipy-free IC/RankIC already written there). → Done 2026-10-06. Checked on dji and nasdaq100 y2020 (sl20/pl5, `CLEAN_DATA = False`): the train/valid/test samples, tensors and dates are identical to FinBench's `CSVDataset`, and the stored seed-42 checkpoints reproduce their test predictions (max diff 1e-5, GPU vs CPU) and metrics. `CLEAN_DATA` drops 3.5% of the dji y2020 samples (`PRG.US`, 1 of 29 tickers) and none of nasdaq100's. The training loop itself moves to `training/trainer.py` in Step 3.
- [x] Write an adapter for each of the other five models, copying from `code/finbench/Regression/<MODEL>/` (`train.py`, model files, `dataloader`/`load_dataset`). Keep FinBench's preprocessing for each model exactly (see the normalization column of the FinBench README table). Record every deviation in the adapter docstring.
  - FinBench issues ([`docs/model_notes.md`](docs/model_notes.md)), decided 2026-09-29: **train exactly as FinBench, fix only readout and paths.** Fixed: FactorVAE leakage and random output (1, 2), MATCC market path (5). Kept and documented: FactorVAE `seq_len` label (3), Alpha360 layout (4). Still to do: tell the supervisor, together with the data issues of notebook 01 (*Observations* 3-6) and the data-quality filter.
  - [x] FactorVAE: model code + leak-free, deterministic `predict()` in `xaifin/models/factorvae.py` (tests in `code/tests/`). The adapter wrapper comes with the rest of Step 2.
  - [x] **Alpha158 group** (2026-10-06): `models/base.py` (`RunConfig`, `ModelAdapter`), `MASTERAdapter`, `MATCCAdapter` (+ `models/matcc.py`, `training/lr_scheduler.py`), `FactorVAEAdapter` (+ `data.datasets.factorvae_splits`, Qlib's calendar windows with filled gaps). Checked on dji y2020: FactorVAE's train/valid/test samples identical to FinBench's `TSDataSampler`; MATCC's outputs and learning-rate schedule identical to FinBench's; the MASTER checkpoint gives its stored test metrics through the adapter; 30 training days run for each adapter. Two new FinBench findings: issues 7 (FactorVAE leaves KMID/KLEN unnormalized) and 8 (MASTER/MATCC runs not seeded) in `model_notes.md`.
  - [x] **Alpha360 group** (2026-10-06): `HISTAdapter`, `DiscoverPLFAdapter`, `FinFormerAdapter` (+ `models/hist.py`, `discoverplf.py`, `finformer.py`; `data.datasets.RowDataset`, `hist_splits`, `finformer_splits`). x is `[N, 1, 360]` in the CSV's column order for all three; each adapter applies FinBench's own reshape inside, so their attributions share one input space. Checked on dji y2020: train/valid/test samples, market values, concept rows and FinFormer's graph identical to FinBench's loaders; outputs identical to FinBench's model classes with the same weights; `concordance_cc` identical to audtorch's. All six adapters train for 30 days on the GPU and give finite input gradients in eval mode. New findings: issues 9 (HIST/DiscoverPLF only run on a GPU, fixed) and 10 (training-loop differences), and issue 8 extends to FinFormer.
  - HIST/DiscoverPLF: pass `stock2concept` and `market_value` through `extras`.
  - FinFormer: pass the sector/industry adjacency through `extras`. Its label is a daily cross-sectional z-score, not the CSRank the README describes.
  - Alpha360 models: compute attributions on the **original 360 columns**, then map them to (series, lag).
- [x] `models/registry.py`: `get_adapter("HIST")`. → Also `load_run(run_dir)` (adapter with its checkpoint, from `config.json`, or from the folder names for the old runs without one, trained with the filter off) and `find_runs()`.
- [x] **Equivalence test:** for each model, train briefly on `dji` y2020 with the FinBench script *and* with the adapter using the same seed, then check that predictions and metrics match within tolerance (except where an issue from `model_notes.md` was deliberately fixed). Run it with `CLEAN_DATA = False`, since FinBench keeps the bad prices. → Done 2026-10-06 in **notebook `02_model_adapters.ipynb`** with `notebooks/finbench_reference.py` (FinBench's own data code, models, training-step and evaluation functions, loaded read-only; the scripts themselves can't run here) and the first part of `training/trainer.py` (`train_epoch`, `evaluate`). One full epoch, same initial weights, day order and random state on both sides: all six pass (initial weights identical; weights after one epoch identical or within 4e-5, Adam on rounding-level gradients; test predictions within 2e-6; FactorVAE against FinBench's leak-free `prediction()`). One section per model.
- [x] **App page `2_Models.py`**: pick model / universe / year / seed → load the checkpoint, show config, training curve, test metrics, and the prediction-vs-label scatter for a chosen date. → Tabs: Overview (stored metrics, configuration, hyper-parameters), Training (loss and validation IC/RankIC per epoch), Test predictions (recomputed through the adapter next to the stored metrics, daily RankIC, prediction vs label with the top-k highlighted). Charts in `viz.py`. Checked headless with Streamlit's `AppTest` on both MASTER runs: the recomputed metrics equal the stored ones.

**Done when:** all six adapters pass the equivalence check and `adapter.forward(x, extras)` returns `[N]` on a `DayBatch` for each model.

---

### Step 3: Rolling-window training and checkpoints (Outline Step 2, *Prima parte* 5)

**Decision (2026-10-06): Option A, every model keeps FinBench's own training and selection rule** (`docs/model_notes.md` issue 10), so the results stay comparable with FinBench's tables: MASTER stops when the training loss falls below 0.95 and keeps the last weights; MATCC keeps the last of 70 epochs; FactorVAE keeps the lowest validation loss; HIST and DiscoverPLF keep the best validation IC of the weights averaged over the last 5 epochs (DiscoverPLF from epoch 20), with early stopping; FinFormer keeps the best validation IC pooled over all stock-days, with early stopping. The rule is `hparams["selection"]` of each adapter. Every run still logs the validation metrics of every epoch (`train_history.csv`), so other rules can be studied later without retraining.

#### Tasks
- [x] `training/trainer.py`: a generic loop with ~~early stopping on valid RankIC~~ each model's FinBench selection rule (`fit`, `SELECTION_RULES`), plus `train_epoch`, `evaluate`, `validation_loss`, `average_params`.
- [x] `training/rolling.py` + `scripts/train_rolling.py --models MASTER --universes dji --years 2020-2024 --seeds 0 5 42 --sl 20 --pl 5`. Skips runs that already exist (resumable); `--models all`; logs to `results/Regression/train_rolling.log`. Each run writes `model.pth`, `config.json` (with a `training` section: rule, kept epoch, validation metrics, time, device), `metrics.json`, `train_history.csv` and FinBench's `results_sl<T>_pl<L>.pkl`.
- [x] Migrate the existing MASTER results to the new layout (done 2026-09-29, see §2.2). → Moved to `results/legacy/` on 2026-10-06: they used another selection rule and no data-quality filter.
- [x] FinBench's published numbers: `results/analysis/finbench_table2.csv` (Table 2: MSE and MAE only; FinBench reports no IC for these six models).
- [ ] **Extended grid on the external H200** (2026-10-22): sp500 and sxxp added, horizons L = 1, 5, 20, so 1242 runs (117 done). **All three horizons are required.** Launched as one grid (the scheduler packs big and small tasks together); ~13 days, so back around 4–6 Nov: this is the critical path of the calendar. When it is back: `grid.py status` 1242/1242, `export_results.py`, `run_portfolio.py`, notebooks 03 and 04 re-run.
- [x] Run the grid **in stages**, checking each stage before going on:
  1. MASTER × dji × y2020 × seed 42 (already done, re-run through the package); → ✅ 2026-10-06: stopped at epoch 6 (training loss 0.947), test IC 0.044, RankIC 0.053, MSE 0.977; 1 minute on the RTX 5090;
  2. all 6 models × dji × y2020 × seed 42 → compare with the FinBench tables (`code/finbench/results.md`). FactorVAE will score lower if its test-time leakage is fixed; → ✅ 2026-10-06. Test RankIC 0.009 (DiscoverPLF) to 0.053 (MASTER); MSE in FinBench's range for HIST (0.963 vs 0.997) and DiscoverPLF (1.176 vs 1.211), higher for MATCC (1.60 vs 1.13) and FinFormer (2.93 vs 1.66, scale-free loss), lower for MASTER (0.98 vs 1.25); one year and one seed against FinBench's averages, so indicative only. Validation IC at the kept epoch is often ≤ 0: the selection rules pick on noise. FactorVAE keeps epoch 1 of 30 (lowest validation VAE loss). Times: MASTER 1 min, FinFormer 3, HIST 6, MATCC 13, DiscoverPLF 17, FactorVAE 50;
  3. all 6 models × {dji, nasdaq100, sx5e} × 2020–2024 × seed 42. SX5E only has 2021–2024: its membership history starts on 2020-09-30, so the 2020 universe is empty (notebook 01); → ✅ 2026-10-07: 84 runs, no failures, about 29 GPU hours (FactorVAE 15 of them). Results in notebook 03;
  4. add seeds 0 and 5. The full grid is 6 × 3 × 5 × 3 = **270 runs**, so estimate the time per run first. → ✅ 2026-10-09: 168 runs, no failures; the grid has **252 runs** (EURO STOXX 50 has no 2020), about 88 GPU hours in total (FactorVAE 47). Seed noise is about half the model differences in RankIC, and two seeds of one model share only 17-46% of their portfolio (notebooks 03 §5, 04 §5); First timings (dji, GPU): about 8-10 s per epoch for MASTER, MATCC, HIST, FinFormer, 22 s for DiscoverPLF, **100 s for FactorVAE** (its 96 attention layers run one after another for every day), i.e. about 50 minutes per FactorVAE run.
- [x] **Notebook `03_rolling_training.ipynb`** (✅ run on the complete grid, 252 runs, with observations and a seed section: grid coverage and timing, RankIC/IC per model × year per universe, averages vs FinBench's Table 2, how weak the signal is (share of positive RankIC, t-statistic adjusted for overlapping labels), the selection rules (validation vs test IC), `metrics_summary.csv`; final run and observations once stage 3 is complete; results from `training.rolling.collect_results`): tables of IC, RankIC, ICIR and MSE per model × universe × year (mean ± std over seeds). Sanity-check that the numbers are in the FinBench range. This notebook also documents *how weak the signal is*, which is the starting point of the thesis.
- [x] **App page `2_Models.py`** (extended): a metrics grid (heatmap model × year) for the selected universe. → Universe, metric and (T, L) selectors; mean over seeds with counts and standard deviations; `viz.metric_heatmap` (blue-red diverging around 0) and `viz.metric_by_year_chart`. The six models have fixed colors (`viz.MODEL_ORDER`, palette validated for colorblind safety in both modes).

**Done when:** a checkpoint exists for every cell of the chosen grid, and the metrics summary table is saved to `results/analysis/metrics_summary.csv`.
**Thesis artefacts:** model-performance table; IC-per-year plot.

---

### Step 4: Portfolio baseline (long-only top-k)

**Why:** the outline cares about how explanations translate into **allocation**. Steps 7–9 need a reference portfolio for every run.

#### Tasks
- [x] `portfolio/topk.py`: on each **rebalance date** (every L=5 business days, as FinBench does with `freq=f'{pl}B'`), pick the top-k stocks by predicted score with equal weights. Store `score`, `rank` and the **margin to the k-boundary** (`score_i − score_(k+1)` for held stocks, `score_k − score_i` for excluded ones). Step 9 builds on this margin. → Done 2026-10-06: `load_predictions`, `rebalance_periods`, `topk_holdings` (every candidate stock kept, held or not, with score, rank, margin and `margin_z` = margin in standard deviations of the day's scores, comparable across models), `overlap`.
- [x] `portfolio/backtest.py`: daily returns from `adj_close`, then CAGR, Sharpe, Sortino, MaxDD, annualized volatility and **turnover**. Use FinBench `Evaluation/portfolio/` (`transforms.create_long_short_portfolio_history`, `returns.portfolio_daily_returns`) as the reference implementation and reproduce its numbers for one run. Benchmark: the equal-weight universe and the index. → Done: `asset_returns`, `portfolio_returns` (buy at the open on the first day, then close-to-close, as FinBench), `performance` (quantstats formulas), `turnover`, `yearly_returns`, benchmarks (equal weight of all candidates; the index from the market file), `run_portfolio`, `run_portfolio_grid`, `find_portfolios`. **Matches FinBench exactly** on MASTER dji 2020-2024: same holdings in all 260 periods, daily returns within 3e-17, identical CAGR/Sharpe/Sortino/drawdown/volatility vs its quantstats (notebook 04 §1).
- [x] **Seed-instability metric:** for the same model/year, the Jaccard overlap of top-k holdings across seeds and the turnover between seed portfolios. This is the first number that shows "small prediction changes → different portfolio". → `topk.overlap` (Jaccard and turnover between two portfolios), used across seeds and across models in notebook 04 §4-5; filled in as seeds and models finish.
- [x] `scripts/run_portfolio.py` for the grid. → Done: every (model, universe, seed) whose test years are all trained (`--partial` otherwise); skips existing.
- [x] **Notebook `04_portfolio_baseline.ipynb`**: equity curves per model, a CAGR/Sharpe table, the holdings overlap across seeds and models, and the distribution of boundary margins. → Run on all 54 portfolios (3 seeds), with observations, including the overlap and return spread between seeds.
- [x] **App page `6_Portfolio.py`** (first version): equity curves, a holdings heatmap (date × ticker), and the margin histogram. → Done: equity curves of the selected models vs the benchmarks, metrics and yearly returns, holdings map, margin histogram, one period's candidates. Checked headless.

**Done when:** `portfolio/…/metrics.json` exists for every trained run and one run matches FinBench's evaluation output.
**Thesis artefacts:** long-only top-k table; seed-overlap figure.

---

### Step 5: XAI engine: attributions (Outline Step 2, *Prima parte* 4)

**Goal:** for every (model, universe, year, seed), produce attributions of the prediction to the inputs, save them, and rank features.

#### 5.1 Methods to implement (`xaifin/xai/`)

| Family | Method | Resolution | Notes |
|---|---|---|---|
| Gradient (Captum) | Saliency, **Integrated Gradients**, GradientSHAP, DeepLIFT | full `[T,F]` per stock | fast on GPU; main workhorse. Baseline = 0 (training median) |
| Shapley / surrogate | **KernelSHAP / PermutationSHAP**, **LIME** | **feature groups** | full `T×F` (e.g. 20×221) is too many players, so mask a feature across all timesteps (F players), or use families (~35 players). **Deadline scope: dji y2020 only**, as a check of IG |
| Global, model-agnostic | **Cross-sectional permutation importance** | per feature / family | shuffle a feature *across stocks within each day*, then measure the drop in RankIC and the change in top-k membership. The metric closest to finance |
| Time-series specific | lag-window occlusion, TimeSHAP-style (feature × time-step × event) | per lag / window | answers "which days in the lookback matter?" |
| Intrinsic | attention weights (MASTER T/S attention, gate; MATCC; FinFormer), HIST concept weights, FactorVAE factor exposures | model-specific | the MASTER notebook already notes that the attention modules must be patched to *return* their weights. Use this as supporting evidence only (attention ≠ explanation) |
| Reference | TreeSHAP on LightGBM | exact | anchors "what the data allows" (Step 7). *(cut unless time is left)* |

- [ ] `xai/base.py`: an `Explainer` protocol, `explain(adapter, batch, target) -> attr[N,T,F]`. `target` is either `"pred"` (the raw score) or `"topk_margin"` (the score minus the k-boundary score, which explains *portfolio membership* rather than the score).
- [ ] Implement the methods above, one module per family.
- [ ] Explain only on **rebalance dates** (≈50 per year) to keep storage manageable: 50 × 100 stocks × 20 × 221 × 2 bytes ≈ 45 MB per run in float16. Use the same L = 5 dates for every horizon (L = 1 would have five times as many).
- [ ] First, measure the cost of IG on one run (MASTER and FactorVAE, dji y2020) to size the batch over the 252 runs. Permutation importance by **family** (~35 groups), not by single feature.

#### 5.2 Aggregation and ranking (`xai/aggregate.py`)
- [ ] Normalize each sample: `ã = |a| / Σ|a|`, so that models with different output scales are comparable.
- [ ] Global importance per **feature** (sum over T), per **lag** (sum over F), per **family** and per **semantic family**. Keep the mean and the dispersion over dates/stocks. Also keep the *signed* mean, to see the direction of the effect.
- [ ] Rank the features for each run. Save `global.parquet`.

#### 5.3 Batch runner
- [ ] `scripts/explain.py --method ig --model MASTER --universe dji --years 2020-2024 --seeds 0 5 42`.

#### 5.4 Notebooks and app
- [ ] **Notebook `05_xai_single_model.ipynb`**: MASTER × dji × y2020. Run every method, then show a `T×F` heatmap for one stock/day, the top-20 features globally, a family bar chart, the lag profile, permutation importance vs IG, and attention maps. Only after this works, generalize to the other models.
- [ ] **App page `3_Explain.py`**: pick model/universe/year/seed/method → (a) global ranking bar chart plus family breakdown, (b) a date + ticker selector showing the `T×F` heatmap and a waterfall of the top features, (c) an "explain now" button that runs a gradient method on demand for one day (GPU, cached), (d) the attention viewer for models that expose it.

**Done when:** `global.parquet` exists for IG and permutation importance for every available L = 5 run on dji, nasdaq100, sx5e (117 now, all three seeds once the H200 grid is back), for both targets (`pred` and `topk_margin`), and SHAP/LIME exist for the stage-2 grid (every model, dji, y2020). If a model's internals (HIST graph, FactorVAE, attention weights) take more than a day, keep IG and permutation for it and skip its attention.
**Thesis artefacts:** top-feature tables per model; family-importance figure; example local explanation.

---

### Step 6: Evaluating the explanations

**Why:** with weak models, explanations can be noisy too. The explanations have to be trustworthy before any conclusion is drawn from them (*Prima parte* 4: "salvare risultati ed eventuali metriche di valutazione").

#### Tasks (`xai/evaluation.py`, results in `xai_metrics.json`)
- [ ] **Faithfulness:** deletion/insertion curves. Mask the top-k attributed features (with the baseline), record |Δ prediction| and ΔRankIC, and compare with random masking. Report AOPC.
- [ ] *(cut)* **Stability / robustness:** similarity of attributions under small input noise (max-sensitivity). Also similarity between neighbouring days for the same stock.
- [ ] **Sanity check** (Adebayo et al. 2018): randomize the model weights; the attributions must change.
- [ ] **Method agreement:** rank correlation between IG, SHAP, LIME and permutation importance on the same run.
- [ ] *(cut)* **Baseline sensitivity:** zero baseline vs per-day cross-sectional mean.
- [ ] **Notebook `06_xai_evaluation.ipynb`**: pick the methods to trust for the rest of the thesis, and justify the choice.
- [ ] *(cut)* **App:** a "quality" tab on `3_Explain.py` with the deletion curve and the method-agreement heatmap.

**Done when:** a documented choice of primary method(s) with the evidence behind it.

---

### Step 7: Cross-model comparison and the data-vs-model diagnosis (Outline Steps 2 and 3)

**Goal:** rank and compare features across models, check whether some features consistently dominate, and decide whether the instability comes from the **data** or the **architecture**.

#### Tasks
- [ ] `analysis/agreement.py`: Spearman ρ, Kendall τ, top-k Jaccard and **rank-biased overlap** between importance rankings.
  - Within group A (and within B): at the feature level.
  - Across groups: at the semantic-family / horizon-bucket level (§Step 1).
- [ ] `analysis/variance.py`: collect the importance vectors indexed by (model, seed, year, universe) and compare the agreement:
  - **same model, different seeds** (training noise),
  - **different models, same seed/year** (architecture),
  - **same model, different years** (non-stationarity of the data),
  - **same model, different universes** (universe effect).
  A variance decomposition (ANOVA-style) over these factors gives a single table.
- [ ] **Diagnosis logic** (write it explicitly in the thesis):
  - models **agree** on features, and IC is low for all of them → the signal in the data is limited or noisy (**data-driven instability**);
  - models **disagree** while having similar IC → several equally good but different explanations exist (Rashomon effect), so architecture drives *which* signal is used (**model-driven**);
  - seed disagreement ≈ model disagreement → the explanations themselves are not stable, so no architectural conclusion can be drawn;
  - LightGBM/TreeSHAP reference *(cut unless time is left)*: if the deep models' top features match it, the data sets the pattern.
- [ ] Link to allocation: repeat the comparison with `target="topk_margin"` attributions. Do the features that drive the *portfolio* differ from those that drive the *score*?
- [ ] **Notebook `07_cross_model_comparison.ipynb`**: agreement heatmaps (model × model), seed-vs-model agreement boxplots, the variance-decomposition table, and "consistently dominant features" (features in the top-10 across ≥ X% of runs).
- [ ] **App page `4_Compare.py`**: multi-select models → side-by-side rankings, an agreement heatmap, a family radar/bar chart, and seed-dispersion error bars.

**Done when:** the variance-decomposition table exists and a written diagnosis is backed by it.
**Thesis artefacts:** the core results chapter: agreement heatmaps, dominant-feature table, diagnosis.

---

### Step 8: Period and regime analysis over rolling windows (*Prima parte* 5)

**Goal:** find out whether particular periods (high volatility) change model behaviour and explanations.

**Deadline scope (light):** yearly only, 2020 and 2022 against the calmer years: IC, portfolio returns and family shares per year, plus the year-to-year drift. The monthly series and the app page are cut.

#### Tasks
- [ ] `analysis/regimes.py`: realized volatility of the index from `<nation>_market.csv` (e.g. `std_price_change_20_*`), then label each date/month low/medium/high volatility. Mark the known events: 2020 COVID crash, 2022 rate-hike bear market.
- [ ] Per regime and per year: IC/RankIC, portfolio returns, **importance drift** (the distance between importance vectors of consecutive windows), and family shares (e.g. does volatility/market importance rise in turbulent periods?).
- [ ] *(cut)* A monthly (not yearly) attribution time-series inside each test year.
- [ ] **Notebook `08_period_analysis.ipynb`**: a timeline plot (index vol, IC and family shares stacked on shared x-axes), and a regime × model table.
- [ ] *(cut)* **App page `5_Over_time.py`**: pick model/universe → importance over time with volatility shading, plus a year-to-year drift chart.

**Done when:** a figure and a short text answering "are some periods more unstable, and do the explanations shift with them?".

---

### Step 9 (optional): Counterfactual stability of the portfolio (Outline Step 4)

**Goal:** measure how much inputs or predictions must change to alter the portfolio's composition.

#### 9.1 Prediction space (cheap, do first)
- [ ] Use the margins from Step 4: the minimal |Δŷ| to flip each stock in or out of the top-k.
- [ ] Compare the margins with the **seed standard deviation of predictions** (Step 3). *Decision fragility* = the fraction of holdings whose margin is smaller than the seed noise.
- [ ] Flip rate under Gaussian noise ε on predictions, for ε in a grid; the resulting turnover and return change.

**Deadline scope:** 9.1 only (the margins exist since Step 4), in notebook 04 or a short notebook 09. 9.2 is cut and goes to future work.

#### 9.2 Input space (with realistic constraints) *(cut)*
- [ ] **(A) Feature space, gradient-based** (Wachter-style): minimize `‖δ‖₁ + λ·hinge(margin(x+δ))` using `adapter.forward`. Constraints: stay inside the clipped z-range [−3, 3], perturb only mutable features (not `market`), keep perturbations sparse, and move correlated features together (clusters from Step 1).
- [ ] **(B) OHLCV space, black-box** (most realistic, because the features are derived): perturb the raw price/volume path of the last few days (e.g. close ±x%), **recompute Alpha158/360** with the pipeline in `finbench/Evaluation/features/`, re-normalize, and re-predict. Bisection/random search on x finds the minimal perturbation that flips membership.
- [ ] Metrics: distance of the counterfactual (L1 in z-space, number of features changed, % price move), flip rate at a given ε, and which features the counterfactuals change most (compare with Step 5 attributions).
- [ ] `portfolio/counterfactual.py`, **notebook `09_counterfactual_stability.ipynb`**.
- [ ] **App page `6_Portfolio.py`** (extended): select date + stock → margin, the minimal counterfactual, and a "what-if" slider on price perturbation that shows the new top-k.

**Done when:** a distribution of minimal flip perturbations per model, plus a fragility comparison across models.

---

### Step 10: Data-perturbation sensitivity tests (*Seconda parte*) *(cut: future work)*

Not done before the 2026-11-18 deadline; described in the conclusions as future work. The family ablation would also need new training.

These are to be defined with the supervisor based on the results. Scaffold in `xaifin/perturb/` so they plug into the same pipeline:
- [ ] Noise injection on OHLCV at several σ → recompute features → Δ IC, Δ portfolio, Δ attributions.
- [ ] Feature-family ablation: drop a family and **retrain**, a stronger test of importance than masking.
- [ ] Time shift / delayed data, missing days, and universe swap (train on one, test on another).
- [ ] **Notebook `10_perturbation_tests.ipynb`**, and an app tab on `4_Compare.py` or a separate page.

---

### Step 11: Graphical application (Conclusion of the outline)

> *"The expected outcome is a graphical interface for loading models, visualizing feature attributions, and comparing explanations across architectures."*

**Stack:** **Streamlit** multipage app (pure Python, uses `xaifin` directly, supports caching and GPU on-demand explanations). Run with `streamlit run code/app/Home.py`.

The pages are added step by step, so the app grows together with the research:

| Page | Built in step | Content |
|---|---|---|
| `Home.py` | 11 | project summary, results-store status (which runs, attributions and portfolios exist) |
| `1_Data.py` | 1 | universe coverage, features and families, correlations, volatility |
| `2_Models.py` | 2–3 | load a checkpoint, config, training curve, metrics grid |
| `3_Explain.py` | 5–6 | global and local attributions, heatmaps, on-demand IG, attention, explanation quality |
| `4_Compare.py` | 7, 10 | cross-model/seed agreement, dominant features, variance decomposition |
| `5_Over_time.py` *(cut)* | 8 | importance drift vs volatility regimes |
| `6_Portfolio.py` | 4, 9 | top-k holdings, margins, counterfactual what-if |

#### Tasks
- [ ] `app/Home.py` plus a shared sidebar (universe, model, seed, year, method) stored in `st.session_state`.
- [ ] Use `@st.cache_resource` for loaded models and `@st.cache_data` for results-store reads.
- [ ] Every chart function lives in `xaifin` (e.g. `xaifin/viz.py`) so the notebooks and the app draw identical figures.
- [ ] "Upload/choose a checkpoint" flow: pick a run directory → the adapter comes from `config.json` → ready to explain.
- [x] Short README section on how to launch the app.

**Done when:** a new user can start the app, load any trained model, see its attributions, and compare it with another architecture without touching code.

---

### Step 12: Thesis writing hooks

| Chapter (`latex-thesis/chapters/`) | Fed by |
|---|---|
| Introduction | outline objective, the "weak signal → unstable allocation" motivation (Step 3/4 numbers) |
| Background & related work | Step 0.2 |
| Methodology: data, models, rolling protocol, XAI methods, evaluation | Steps 1, 2, 3, 5, 6 |
| Results: performance, attributions, cross-model comparison, periods, counterfactuals | Steps 3–9 |
| The application | Step 11 (screenshots) |
| Conclusions: data vs model diagnosis | Step 7 |

Export every thesis figure from the notebooks to `latex-thesis/images/results/` with fixed filenames, so re-running a notebook updates the thesis.

---

### 4. Progress tracker (updated 2026-11-18)

| Step | Notebook | Package | App page | Status |
|---|---|---|---|---|
| 0 Environment & reading | `00_environment_check` | `pyproject.toml` | – | ✅ done (model notes + chapter 2 drafted; read the papers yourself too) |
| 1 Groups & universes | `01_universes_and_data` | `config`, `data/features` | `1_Data` | ✅ done |
| 2 Adapters | `02_model_adapters` | `data/*`, `models/*` | `2_Models` | ✅ done: data pipelines, six adapters (equivalence test: all pass), registry, app page |
| 3 Rolling training | `03_rolling_training` | `training/*`, `scripts/train_rolling` | `2_Models` | ✅ done: full grid, 1242 runs (5 universes × L = 1/5/20 × 3 seeds; 1125 on the H200, 117 retrained on the RTX 5090), notebook 03 with observations |
| 4 Portfolio baseline | `04_portfolio_baseline` | `portfolio/topk`, `backtest` | `6_Portfolio` | ✅ done: portfolios of the full grid, checked against FinBench, notebook with observations |
| 5 XAI engine | `05_xai_single_model` | `xai/*`, `scripts/explain` | `3_Explain` | ◐ core universes done (756/756 runs, every rebalance date, 6 methods); S&P 500 and STOXX Europe 600 on the H200 (S1–S4); app page missing |
| 6 XAI evaluation | `06_xai_evaluation` | `xai/evaluation`, `scripts/evaluate` | `3_Explain` (quality) | ✅ done on the core (54 runs, R1c; notebook 06; thesis §5.7) |
| 7 Cross-model diagnosis | `07_cross_model_comparison` | `analysis/agreement`, `variance` | `4_Compare` | ✅ done on the core, with the horizon factor (R5); all universes at S4; app page missing |
| 8 Period analysis | `08_periods_and_regimes` | `data/summary` (volatility) | ~~`5_Over_time`~~ | ✅ done (light), all three horizons |
| 9 Counterfactuals (opt.) | `09_decision_fragility` | `portfolio/fragility` | `6_Portfolio` | ✅ 9.1 done (three seeds); 9.2 cut |
| 10 Perturbation tests | – | – | – | ✂ cut: future work |
| 11 Application | – | `viz.py` | `Home` + all pages | ◐ `Home`, `1_Data`, `2_Models`; README explains how to launch it |
| 12 Thesis | – | – | – | ◐ chapters 1–9 drafted; see «Stato al 18/11» |

## 10. Appendice: storico

Parti superate, tenute per riferimento.

### Intestazione e istruzioni originali di PIANO.md (22/10/2026)

Sostituite dalla sezione 1.

> Documento di progetto per il periodo in cui la griglia di training gira sulla H200 dell'amico.
> Il lavoro descritto qui si fa sulla **macchina con CUDA** (quella degli stage 1–4).
> Il piano tecnico completo resta THESIS_GUIDE.md; qui ci sono lo stato reale,
> cosa fare, le decisioni prese e il registro. Aggiornato il 2026-10-22.
>
> Per Claude (o chi riprende il lavoro):
> 1. Leggi questo file, poi le parti della guida citate (Step 5–9, 11), poi `docs/model_notes.md` per i dettagli dei modelli.
> 2. Fai le voci della sezione "Lavoro" **in ordine**, spuntandole qui (`- [x]`) con una riga su cosa è stato fatto. Le decisioni nuove vanno nel registro in fondo, con la data.
> 3. Regole: non modificare modelli, iperparametri, seed, protocollo o regole di selezione: lo XAI spiega i checkpoint così come sono; non cancellare e non sovrascrivere nulla in `code/results/Regression/` (`--force` mai); lo XAI scrive solo nelle sue cartelle (§2.2 della guida); le figure per la tesi vanno in `latex-thesis/images/results/` con nomi fissi; la tesi è **in inglese**; codice e commenti nello stile del repo.
> 4. Se una scelta non è coperta da questo file o dalla guida e cambia i risultati, chiedi prima.

### Stato reale al 22/10/2026 (da PIANO.md)

| Cosa | Stato |
|---|---|
| Griglia sulla H200 | avviata dall'amico con `launch_grid.py` (orchestratore scritto dal suo Claude): 1242 run, L = 1/5/20, 5 universi, 3 seed. ETA ~13 giorni (12–17) → rientro **~4–9 novembre**. Stato sul suo server: `grid.py status` e `_job/claude/status.json`. Risultati con `code/scripts/remote/pack_results.sh` |
| Run finite disponibili | **117** (L = 5): seed 42 per tutti e 6 i modelli su dji, nasdaq100, sx5e (84 run), MASTER anche con seed 0 e 5, MATCC e FactorVAE con qualche seed 0/5 sparso. **Le run seed 0/5 dello stage 4 sono perse**: arriveranno dalla griglia H200 |
| Checkpoint | `model.pth` è in `.gitignore`: non passa con git. Sulla macchina CUDA verificare con `grid.py status` che ci siano le 117 run; se mancano, trasferire lo zip di `export_results.py --zip` |
| Ambiente | la guida §0.1 dà captum, shap e lime installati sulla macchina CUDA: verificarlo con `00_environment_check.ipynb` (il PC Windows ha torch solo CPU e non li ha) |
| Tesi (`latex-thesis/`, in inglese) | cap. 2 Background ~3700 parole (bozza); introduzione ~60; cap. 3 è un segnaposto ("Tables and Images"); conclusioni vuote; mancano metodologia e risultati |
| Codice | Step 0–4 fatti; `xai/`, `analysis/`, `perturb/` vuoti; app con `Home`, `1_Data`, `2_Models`, `6_Portfolio` |

### Calendari superati (da PIANO.md)

| Date | Lavoro |
|---|---|
| 22–24 ott | A1, A2, B1, B2 su MASTER; F1 |
| 25 ott – 1 nov | B3–B7 (batch sulle 117 run); F3 |
| 2–6 nov | C1, C2 (seed solo MASTER); D1; F2 sui numeri attuali |
| ~4–9 nov | rientro griglia → D2; C2 esteso ai seed, C3, C4 |
| **~10 nov** | **bozza al relatore**: metodologia, prestazioni, primi risultati XAI |
| 10–17 nov | E1, E2; F4, F5; correzioni del relatore |
| 18 nov | consegna (scadenza formale del file: 19 nov) |

Calendario rivisto il 2026-11-05 (rientro griglia e Step 5 fatti, D2 sezioni A e B chiuse):

| Date | Lavoro |
|---|---|
| 5 nov sera | fine batch XAI seed 42; C2 (Shapley/LIME dji 2020) |
| 6 nov | notebook 05, 06, 07, 08 rieseguiti; osservazioni; scelta dei metodi (Step 6), diagnosis (Step 7) |
| 7–8 nov | tesi: cap. 5 §valutazione, cap. 6 §6.2–6.6, risposte alle RQ (cap. 9), TODO dell'introduzione |
| 9 nov | rilettura, compilazione LaTeX, figure mancanti |
| **10 nov** | **bozza completa al relatore** |
| 10–14 nov | app (`3_Explain`, `4_Compare`) e cap. 8; copia dei risultati sul drive |
| 14–17 nov | correzioni del relatore |
| **18 nov** | **consegna** (margine: 19 nov) |

Aggiornamento 2026-11-05: le bozze sono scritte in prosa rifinita; Luca le rivede e le riscrive a mano dove vuole
(~24.000 parole, 8–12 giorni). Quindi:
riscrittura dei capitoli stabili (2, 3, 4, 7) dal 6/11 mentre si chiudono 5, 6, 8, 9; bozza dei contenuti al
relatore il 10/11; riscrittura di 5, 6, 8, 9 e introduzione il 10–14/11; correzioni 14–17/11; consegna 18/11.
Seduta di laurea obiettivo: 1/12/2026; caricamento su Esse3 entro il 19/11.

### Scadenza del 18/11 e calendario della guida tecnica (superati)

> **Deadline: everything finished by 2026-11-18** (decided 2026-10-22). The full plan below would take about two months, so it is cut to fit:
> - **XAI starts on the 117 runs available** (seed 42 of all six models on dji, nasdaq100, sx5e × L = 5, plus MASTER seeds 0 and 5; the stage-4 seed 0/5 runs of the other models were lost). The whole grid (all seeds, sp500, sxxp, and the **required** horizons L = 1 and 20) is trained in parallel on an external H200 (`code/scripts/remote/`, ~13 days). Steps 3–4 cover the full grid with all three horizons. XAI then extends to the seed 0/5 runs at L = 5 (needed for the seed factor of Step 7), and, if time is left, to L = 1 and 20 on the same three universes.
> - Working plan, decisions (methods, targets, granularity) and log: the Italian sections at the top of this file.
> - **Kept:** Step 5 with IG, permutation importance by family and attention where easy; Step 6 reduced; **Step 7 in full** (the core result); Step 8 light; Step 9.1; app pages `3_Explain` and `4_Compare`; writing.
> - **Cut (future work in the thesis):** SHAP/LIME beyond dji y2020, Step 9.2, Step 10, page `5_Over_time`, the LightGBM/TreeSHAP reference unless time is left.
> - The calendar is in §3. Items marked *(cut)* below are not done before the deadline.

**Calendar to the 2026-11-18 deadline** (replaces the milestone plan below where they differ):

| Dates | Work | In parallel |
|---|---|---|
| 22–24 Oct | launch the full H200 grid (L = 1, 5, 20); IG cost measured; `xai/base.py` + IG on MASTER × dji × y2020 | writing: methodology (data, models, rolling protocol) |
| 25 Oct – 1 Nov | Step 5: IG, permutation by family, attention where easy, for the 6 models; aggregation; `explain.py` batch on the 117 runs | writing: XAI methods |
| 2–6 Nov | Step 6 reduced; **Step 7 in full** | check the H200 every day |
| 5–9 Nov | grid back → Steps 3–4 re-run on the full grid; Step 8 light; Step 9.1 | |
| 10–17 Nov | results and conclusions chapters, figures, app pages `3_Explain` and `4_Compare` | draft to the supervisor as early as possible |
| 18 Nov | submission | |

| Milestone | Steps | Outcome |
|---|---|---|
| **M1: Foundations** | 0, 1, 2 | environment fixed, package skeleton, 6 adapters verified |
| **M2: Checkpoints** | 3, 4 | rolling grid trained (stage 3), portfolios computed |
| **M3: Explanations** | 5, 6 | attributions for the grid + validated choice of methods |
| **M4: Analysis** | 7, 8 | cross-model diagnosis, period analysis → core results |
| **M5: Extensions** | 9, 10 | counterfactual stability, perturbation tests |
| **M6: Application & write-up** | 11, 12 | GUI complete, thesis chapters |

Build the app page for each step **in the same milestone** as the step, not at the end.

---

### Rientro della griglia e primo batch XAI (da RIESECUZIONE.md, RA–RC)

Promemoria operativo per chiudere il rientro della griglia e lo Step 5, scritto il 2026-11-04 mentre girano
il riaddestramento delle 117 run e il batch XAI. Il perché di ogni passo è nel registro delle decisioni (sezione 4).
Spunta le voci (`- [x]`) man mano.

Tutti i comandi vanno lanciati dalla radice del repository, con l'ambiente attivo (`.venv\Scripts\activate`).

#### I job del 4–5/11

| Job | Log | Finito quando |
|---|---|---|
| Riaddestramento delle 117 run pre-griglia (RTX 5090) | `logs/grid_retrain_117.out`, `logs/grid/task*.log` | `python code/scripts/grid.py status` dice **1242/1242** |
| Batch XAI, seed 0/5 a L = 5 (3 processi) | `logs/xai_a.out`, `logs/xai_b.out`, `logs/xai_c.out` | l'ultima riga di ciascuno è `[N/N] ...` (a: 28, b: 56, c: 51 run) e il processo python non c'è più |
| Ricalcolo portafogli (regola sulle aperture) | `logs/portfolio_fix.out` | ✅ finito il 2026-11-04 ("ALL DONE") |

Errori: `grep -l Traceback logs/grid/*.log` e `grep Traceback logs/xai_*.err`.

#### RA. Quando finisce il riaddestramento (1242/1242)

Stima: domani 5/11, pomeriggio o sera.

- [x] **RA1. Verifica della riproduzione** → 5/11 11:45: 99/117 identiche; le 18 diverse sono tutte e sole le MATCC
  (RankIC ±0,008 in media, max 0,024; MSE ±0,016; un cambio di segno su 18). Registro delle decisioni.
  `python code/scripts/check_reproduction.py`
  Atteso: `117/117 identical`, **tranne le run MATCC** (non deterministiche anche sulla stessa GPU, vedi registro
  2026-11-05): per MATCC registrare quante differiscono e di quanto, e usare le nuove. Se qualche run è "different", **fermarsi**: i numeri dei seed 42 già nei
  notebook e nella tesi andrebbero rivisti (scrivere nel registro delle decisioni).
- [x] **RA2. Portafogli delle run riaddestrate** (5/11: L = 5 core ricalcolati con k di default e k = 5; 432
  portafogli con parquet, nessun rendimento giornaliero > 50%)
  ```
  python code/scripts/run_portfolio.py
  python code/scripts/run_portfolio.py --k 5 --universes dji nasdaq100 sx5e
  python code/scripts/run_portfolio.py --universes dji nasdaq100 sx5e --force
  ```
  Il terzo comando serve ai 18 portafogli seed 42 a L = 5 che hanno solo i CSV (senza parquet): li
  ricalcola uguali. Se si vuole limitare il `--force` al solo L = 5, va bene anche rifare tutto: dji e sx5e
  non cambiano con la regola delle aperture, nasdaq100 è già ricalcolato.
  Controllo: nessun portafoglio con rendimenti giornalieri > 50% (il controllo fatto il 4/11 è nel registro delle decisioni).
- [x] **RA3. Notebook 03** (5/11: rieseguito, 1242 run in `metrics_summary.csv`, Observations riscritte) (`code/notebooks/03_rolling_training.ipynb`): rieseguire tutto. Rigenera
  `code/results/analysis/metrics_summary.csv` con tutte le 1242 run. Riscrivere le **Observations**: i seed 0/5
  a L = 5 sono quelli della H200 (diversi dagli originali persi), più la nuova sezione 7 sugli orizzonti.
- [x] **RA4. Notebook 04** (5/11: rieseguito, Observations riscritte) (`04_portfolio_baseline.ipynb`): rieseguire tutto (rigenera `portfolio_summary.csv`,
  `portfolio_margins.csv`, `portfolio_horizons.csv`). Riscrivere le **Observations** (nuova sezione 6 sugli
  orizzonti; a L = 1 portafogli ed equal weight perdono il rendimento notturno, vedi registro).
- [x] **RA4b. Notebook 09** (5/11: rieseguito con i tre seed, Observations scritte) (`09_decision_fragility.ipynb`, Step 9.1): rieseguire con i tre seed (salva
  `results/analysis/decision_fragility.csv`) e scrivere le **Observations**.
- [x] **RA5. Confronto con FinBench** (5/11: tabelle principali a L = 5 e complete per l'appendice `*_horizons.tex`;
  portafogli solo sugli universi della Table 7): `python code/scripts/compare_finbench.py` (dopo RA3, perché legge
  `metrics_summary.csv`). Scrive `latex-thesis/tables/finbench_regression.tex` e `finbench_portfolio.tex`
  con i tre orizzonti.
- [x] **RA6. Tesi, `04_framework.tex`** (5/11: paragrafi su training, riproducibilità, confronto FinBench su tre
  orizzonti e Table 7, riassunto; nuova appendice `98_appendix.tex` con le tabelle per orizzonte):
  - riscrivere il paragrafo "252 runs ... 88 hours on a single RTX 5090": 117 run sulla RTX 5090, 1125 sulla
    H200 NVL (~12 giorni, ~919 ore-GPU sommate sulle run); training riproducibile sulla stessa GPU (RA1), non tra
    GPU diverse (i seed 0/5 rifatti sulla H200);
  - aggiornare il testo sulla Table finbench_regression ("Of the 18 model--universe pairs ..."), ora con tre
    orizzonti;
  - chiudere il TODO sul confronto con la Table 7 (`finbench_portfolio.tex`).
- [x] **RA7. Tesi, cap. 7 (portafogli)** (5/11: protocollo e sezione dei risultati con le tabelle `pf_performance`
  e `pf_fragility` da `thesis_figures.py`): regola sulle aperture inaffidabili (SBNY) e l'effetto apertura a L = 1.

#### RB. Quando finiscono i 3 processi del batch XAI

Stima: tra stanotte (processo c) e domani sera (processo a). Non serve aspettare A.

Stato: processo c (FactorVAE, MATCC, MASTER) finito il 4/11 alle ~23:00; per i suoi modelli RB1 e RB2 sono già
fatti (`--reaggregate --models FactorVAE MATCC MASTER`, `--methods intrinsic --models MATCC MASTER`). Restano
DiscoverPLF (processo a). Processo b (FinFormer, HIST) finito il 5/11 alle ~07:40: RB1 fatto
(`--reaggregate --models FinFormer HIST`), RB2 fatto (pesi interni di 28 run FinFormer).

- [x] **RB1. Ricostruire i file globali** (aggiunge la colonna `net`, che i processi partiti il 4/11 non scrivono):
  `python code/scripts/explain.py --reaggregate`
- [x] **RB2. Pesi interni** di MASTER, MATCC, FinFormer (pochi secondi a run):
  `python code/scripts/explain.py --methods intrinsic`

#### RC. Quando RA e RB sono finiti

Stima: 6/11 sera – 7/11 mattina.

- [x] **RC1. XAI sulle run riaddestrate** (seed 42 e i MASTER a L = 5), in 3 processi come il primo batch:
  ```
  python code/scripts/explain.py --models DiscoverPLF
  python code/scripts/explain.py --models FinFormer HIST
  python code/scripts/explain.py --models FactorVAE MATCC MASTER
  ```
  (salta da solo ciò che è già fatto; ora calcola anche `intrinsic` e la colonna `net`). ~12–18 ore.
- [ ] **RC2. Shapley e LIME** (Step 6, controllo di IG), DJI 2020, tutti i modelli:
  `python code/scripts/explain.py --methods shapley lime --universes dji --years 2020`
- [x] **RC3. Notebook 05** (5/11: rieseguito su MASTER dji seed 42 y2020, Observations scritte) (`05_xai_single_model.ipynb`, già impostato su MASTER dji L = 5 seed 42 y2020):
  rieseguire e scrivere le **Observations**.
- [ ] **RC4. Notebook 06** (Step 6): rieseguire; scegliere e motivare il metodo principale e la misura per
  famiglia (`importance` lorda o `net`).
- [x] **RC4b. Notebook 08** (5/11: rieseguito, Observations scritte; accordo per feature senza le feature di mercato) (Step 8, anni e regimi): rieseguire (serve RB1 per le quote nette) e scrivere le
  **Observations**.
- [ ] **RC5. Notebook 07** (5/11: diagnosi scritta; rieseguito alle 17:45 con xai2_a finito, 252/252 run complete; resta da rieseguire dopo il notebook 06 solo se cambiano i numeri del sanity check) (Step 7): rieseguire; scrivere la **diagnosi** dati vs modello. Tenere conto del
  sanity check (HIST e in parte MASTER hanno attribuzioni simili a quelle del modello non allenato).
- [ ] **RC6. Tesi** (5/11: cap. 6 §6.2–6.7 scritti, tabelle `res_inputs`, `res_agreement`, `res_agreement_models`, `res_variance`; cap. 9 apertura e risposte alle RQ; introduzione: risposte in breve e contributo 2. Restano: cap. 5 valutazione e numeri `% TODO` dopo il notebook 06, i due `% TODO` sul sanity check nel cap. 6): sostituire i numeri marcati `% TODO` nei capitoli (`grep -n "TODO" latex-thesis/chapters/*.tex`),
  in particolare `05_xai_methodology.tex`; scrivere la sezione di valutazione (Step 6) e i risultati (cap. 6).
