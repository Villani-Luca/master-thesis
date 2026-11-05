# XAI: come è strutturato, cosa gira dove

Aggiornato il 2026-11-05. Le decisioni e il loro registro sono in [PIANO.md](../PIANO.md); i passi operativi da
spuntare in [RIESECUZIONE.md](../RIESECUZIONE.md) (voci R1–R5 e sezione E).

## 1. A cosa serve

La domanda centrale della tesi è se l'instabilità dei modelli, e dei portafogli costruiti con le loro previsioni,
venga dai **dati** o dal **design dei modelli**. L'XAI serve a vedere *su cosa si basa* ogni modello e a confrontarlo:

- tra **modelli diversi** sugli stessi dati: conta l'architettura?
- tra **seed diversi** dello stesso modello: conta il caso del training?
- tra **anni di test diversi**: conta il periodo di mercato?
- tra **universi e orizzonti diversi**: i risultati si generalizzano?

## 2. Cosa si spiega

| Livello | Cosa | Quanti |
|---|---|---|
| **Run** | Un modello allenato: modello × universo × orizzonte × seed × anno di test | 1242 |
| **Giorni** | Tutte le date di ribilanciamento settimanale del portafoglio nell'anno di test, le stesse per ogni orizzonte | ~52 per run |
| **Titoli** | Ogni titolo dell'universo in quel giorno | da ~30 (DJI) a ~600 (STOXX Europe 600) |
| **Target** | Il **punteggio** del titolo, e il **margine dal confine del top-k** (perché è o non è in portafoglio) | 2 |

Ogni titolo è spiegato rispetto ai **propri** input; quelli degli altri titoli del giorno restano fermi, ma il
modello vede sempre il giorno intero, perché cinque dei sei modelli mettono in relazione i titoli tra loro.

## 3. I metodi

Tutti e sei i metodi girano su **ogni** run, su **tutte** le date, per **entrambi** i target.

| Metodo | Cosa calcola | Ruolo nella tesi |
|---|---|---|
| **Integrated Gradients** (IG) | Il contributo di ogni singolo input (giorno passato × feature) al target. Baseline = il titolo medio del giorno; percorso "joint"; 16 passi | **Metodo principale**, l'unico a piena risoluzione |
| **Saliency** | Il gradiente semplice | Confronto: nelle verifiche non è fedele, e mostra perché serve IG |
| **Shapley esatto per famiglia** | Il contributo di ogni famiglia di feature (7 per Alpha158, 3 per Alpha360), su tutte le 2^P combinazioni; una famiglia "assente" vale il titolo medio | Riferimento teorico: IG aggregato per famiglia (quota netta) deve dire lo stesso |
| **LIME per famiglia** | Approssimazione locale con 200 combinazioni casuali | Secondo controllo, del metodo più usato in letteratura |
| **Permutation importance per famiglia** | Mescola una famiglia tra i titoli del giorno (5 ripetizioni) e misura il calo del RankIC, la quota di portafoglio sostituita e la variazione dei punteggi | Separa ciò che porta **abilità predittiva** da ciò che **muove il portafoglio** |
| **Pesi interni** (solo MASTER, MATCC, FinFormer) | Gate sulle feature, attention nel tempo e tra titoli, gate di fusione | Confronta ciò che il modello "dice" di guardare con ciò che conta davvero |

Più una **valutazione dei metodi** ([notebook 06](../code/notebooks/06_xai_evaluation.ipynb)):

- **sanity check**: le attribuzioni del modello allenato confrontate con quelle dello stesso modello ai pesi
  iniziali. Se sono simili, il metodo mostra l'architettura e non ciò che il modello ha imparato;
- **deletion curve**: togliendo gli input più importanti il punteggio deve calare più che togliendoli a caso;
- **accordo tra metodi** sulla classifica delle famiglie, e verifica delle impostazioni di IG (percorso, baseline).

Oggi la valutazione è su un run per modello (DJI, seed 42, 2020). Estenderla a ogni universo e orizzonte costa
poche ore ed è ancora da decidere.

Codice: [xai/gradient.py](../code/xaifin/xai/gradient.py), [xai/shap_lime.py](../code/xaifin/xai/shap_lime.py),
[xai/permutation.py](../code/xaifin/xai/permutation.py), [xai/intrinsic.py](../code/xaifin/xai/intrinsic.py),
[xai/evaluation.py](../code/xaifin/xai/evaluation.py); script [explain.py](../code/scripts/explain.py).

## 4. Dalle attribuzioni ai risultati

| Passo | Cosa succede | Dove |
|---|---|---|
| 1. Attribuzioni locali | Un valore per titolo, giorno e input (IG, Saliency); uno per titolo, giorno e famiglia (Shapley, LIME) | `attributions_<target>.npz` di ogni run |
| 2. Normalizzazione | Ogni titolo-giorno pesa uguale: quota del totale assoluto delle sue attribuzioni | [xai/aggregate.py](../code/xaifin/xai/aggregate.py) |
| 3. Importanza globale del run | Media su titoli e giorni, a 5 livelli: feature, giorno passato, famiglia, fonte (prezzo, volume, ...) e orizzonte della feature. Per le famiglie, quota **lorda** (somma dei valori assoluti) e **netta** (valore assoluto della somma, quella confrontabile con Shapley) | `global_<target>.parquet` di ogni run |
| 4. Confronti a coppie | Due run che differiscono in **un solo fattore** (seed, modello, anno, universo, orizzonte): quanto concordano le classifiche (Spearman, Kendall, top-10 in comune, RBO) | [analysis/variance.py](../code/xaifin/analysis/variance.py), notebook 07 |
| 5. Scomposizione della varianza | Quanta variabilità delle spiegazioni dipende da ciascun fattore e dalle interazioni (ANOVA di tipo II) | Notebook 07 |
| 6. Periodi e regimi | Spiegazioni negli anni volatili (2020, 2022) contro quelli calmi | Notebook 08 |
| 7. Diagnosi | Dati o modello? | Cap. 6 e 9 della tesi |

Le feature di mercato (uguali per tutti i titoli di un giorno) hanno attribuzione nulla per costruzione con la
baseline del titolo medio, e sono escluse dai confronti; il loro effetto si legge nei pesi interni.

I primi risultati (L = 5, universi core, date campionate) sono già nella tesi. I modelli concordano sul **tipo** di
informazione (le famiglie), non sulle singole feature, e per metà dei modelli il seed cambia la spiegazione quanto
l'architettura. Con la griglia completa la diagnosi si estende ai fattori universo e orizzonte.

## 5. La griglia e chi fa cosa

| | **Questo PC** (RTX 5090) | **Server** (H200) |
|---|---|---|
| Universi | DJI, NASDAQ-100, EURO STOXX 50 | S&P 500, STOXX Europe 600 |
| Orizzonti | L = 1, 5, 20 | L = 1, 5, 20 |
| Run | **756** | **486** |
| Metodi, date, target | tutti | tutti, identici |
| Perché lì | 30–100 titoli: basta questa macchina | 500–600 titoli: ogni giorno costa molto di più, e servono decine di GB di RAM per processo |
| Cosa riceve | — | `C:\Users\villa\Downloads\xaifin_xai_bundle.zip` (codice, 486 checkpoint, guida) ed `E:\Thesis\data.zip` (dati) |
| Analisi e tesi | qui, su tutti i 1242 run | no: manda solo i risultati |

### Come gira

Entrambe le macchine usano lo stesso codice e lo stesso scheduler, [grid.py](../code/scripts/grid.py) con `--xai`:

- divide il lavoro in **task** = modello × universo × orizzonte × seed (4–5 run ciascuno), uno per processo;
- ne fa girare diversi in parallelo; un task parte solo se la sua RAM stimata ci sta;
- comincia da L = 5 e dai modelli più lenti;
- se si interrompe, al rilancio riparte da dove era: i run e i metodi già fatti vengono saltati;
- un run conta come spiegato solo quando ci sono le uscite di **tutti** i metodi.

Comandi utili:

```bash
python code/scripts/grid.py status --xai                                  # avanzamento di tutta la griglia
python code/scripts/grid.py status --xai --universes dji nasdaq100 sx5e   # solo il core
python code/scripts/grid.py run --xai --universes dji nasdaq100 sx5e --workers 5   # lancio sul PC
```

### Il server

Il Claude del server segue `START_HERE.md` dentro lo zip
([sorgente: START_HERE_XAI.md](../code/scripts/remote/START_HERE_XAI.md)):

| Passo | Cosa | Perché |
|---|---|---|
| 1. Ricognizione | GPU, RAM, disco (servono ~400 GB) | Il server può essere condiviso |
| 2. Estrazione | Codice e checkpoint; `data.zip` dentro `master-thesis/code/` | `data.zip` è stato verificato file per file contro i dati di questo PC |
| 3. Ambiente | Python 3.12, torch CUDA, captum | Gli stessi pacchetti di qui |
| 4. Verifica | `grid.py status --xai --universes sp500 sxxp` deve dare 0/486 senza checkpoint mancanti | Pacchetto completo |
| 5. Controllo incrociato | [check_xai.py](../code/scripts/remote/check_xai.py) rispiega MASTER DJI 2020 (53 date, tutti i metodi) e lo confronta con quello di qui | Le due macchine devono dare gli stessi risultati; se no si ferma |
| 6. Prova e misura | Due run veri su S&P 500 (MASTER e HIST): tempo e picco di RAM | Data di fine e numero di processi in parallelo |
| 7. Proposta | Parametri delle risorse, con la conferma di chi lo lancia | |
| 8. Lancio | [xai.sh](../code/scripts/remote/xai.sh), cioè `grid.py run --xai --universes sp500 sxxp` | Prima L = 5 (162 run), poi L = 1 e 20 (324) |
| 9. Invii | [pack_xai.sh](../code/scripts/remote/pack_xai.sh) ogni 2–3 giorni | Risultati da guardare man mano |

## 6. Cosa torna dal server e come si integra

| Pacchetto | Contenuto | Dimensione | Come |
|---|---|---|---|
| **Leggero** (ogni 2–3 giorni) | Importanze globali, Shapley e LIME, permutation, pesi interni, parametri, log, esito del controllo incrociato | 1–2 GB a fine lavoro | Drive o rsync |
| **Completo** (solo se serve) | In più i tensori di ogni singola attribuzione | ~230 GB | rsync o disco fisico; altrimenti resta sul server |

Per integrarlo basta estrarlo dalla radice del repo (`tar -xzf xaifin_xai_light_<...>.tar.gz`). Scrive solo in
`code/results/xai/<metodo>/<modello>/{sp500,sxxp}/` e in `logs/xai_server/`, quindi non tocca nulla del core.
Dopo, `grid.py status --xai` mostra l'avanzamento di tutta la griglia, e i notebook leggono i risultati delle due
macchine insieme.

## 7. Dove sono i file

```
code/results/xai/<metodo>/<MODELLO>/<universo>/sl<T>_pl<L>/seed<S>/y<ANNO>/
    attributions_<target>.npz    # attribuzioni locali (fuori da git)
    global_<target>.parquet      # importanza globale del run
    params_<target>.json         # impostazioni, giorni, righe, tempo
    permutation.parquet, global.parquet   # solo permutation
    weights.npz                  # solo intrinsic (fuori da git)
code/results/xai/explain.log     # una riga per run spiegato
code/results/xai_sampled_dates/  # la prima versione, a date campionate (vedi sotto)
logs/xai/task<N>.log             # log di ogni task dello scheduler
```

`<metodo>` è uno di `ig_joint`, `saliency`, `shapley`, `lime`, `permutation`, `intrinsic`.

## 8. Decisioni prese

| Data | Decisione | Perché |
|---|---|---|
| 4/11 | Baseline = titolo medio del giorno, per tutti i modelli | Con la baseline zero IG non converge per FactorVAE; in un compito di ranking il riferimento naturale è il titolo medio |
| 4/11 | Percorso "joint" di IG | Classifica gli input come il percorso per titolo, a una frazione del costo |
| 4/11 | Spiegare sia il punteggio sia il margine dal top-k | Lega le spiegazioni alle decisioni di portafoglio |
| 5/11 | XAI su **tutti e 5 gli universi e tutti e 3 gli orizzonti** | Richiesta del relatore; laurea spostata al 09/02/2027 |
| 5/11 | **Tutte le date** di ribilanciamento | Prima una data su due a L = 5 e una su quattro a L = 1/20, solo per i tempi della vecchia scadenza. Le uscite di quella versione sono in `code/results/xai_sampled_dates/` e servono a dimostrare che il campionamento cambiava poco: su un run di prova Spearman 0,997 sulle feature, classifica delle famiglie identica |
| 5/11 | **Shapley e LIME su ogni run** | Prima solo DJI 2020, come controllo di IG |
| 5/11 | Universi grandi sulla H200, core su questo PC | Costo e RAM per giorno degli universi grandi |

## 9. Stato e tempi (5/11 sera)

| Parte | Stato | Fine stimata |
|---|---|---|
| Core L = 5 (252 run) | In corso: scheduler partito alle 18:26, prima i task di DiscoverPLF | ~8–9/11 |
| Core L = 1 e 20 (504 run) | In coda dopo L = 5 | ~20–25/11 |
| I primi 5 task del core | Partiti prima che Shapley e LIME diventassero default | Al termine lo scheduler va rilanciato una volta e completa solo quelli |
| Server (486 run) | In attesa dei due zip | Da misurare al passo 6 della guida |
| Notebook 05–08 e numeri dei cap. 6 e 9 | Da rifare quando il core L = 5 è completo | Subito dopo ~8–9/11 |
| Verifica del campionamento delle date (una frase nel cap. 5) | Da fare sul core | Con il core L = 5 |
| Analisi finale su tutti i 1242 run | Dopo l'ultimo pacchetto del server | — |

Ancora da decidere: estendere la valutazione dei metodi (sanity check, deletion curve) a ogni universo e orizzonte.
