# START HERE: spiegazioni (XAI) dei modelli già allenati, su questo server

Guida operativa per Claude Code (o per chi lo fa a mano). Chiedi a Claude:
**"Leggi START_HERE.md e seguilo passo per passo, chiedendomi conferma prima del lancio finale."**

## Contesto

Tesi magistrale (Luca Villani, UNIMORE) su modelli di previsione azionaria e sulla loro spiegabilità.
La griglia di training (1242 run) è già stata allenata, in buona parte su questo server a ottobre. Ora
servono le **spiegazioni** dei run dei due universi grandi, **S&P 500 (`sp500`) e STOXX Europe 600
(`sxxp`)**: 6 modelli × 2 universi × 3 orizzonti (L = 1, 5, 20) × 3 seed × anni di test 2020–2024
(2021–2024 per sxxp) = **486 run**. Gli altri tre universi vengono spiegati sul PC di Luca.

**Non si allena niente.** Per ogni run lo script carica il checkpoint (`model.pth`, già nel pacchetto)
e, sui giorni in cui il portafoglio si ribilancia, calcola:
Integrated Gradients (16 passaggi con gradiente per giorno) e Saliency, sia del punteggio sia del
margine dal portafoglio top-k; valori di Shapley esatti e LIME sulle famiglie di feature (128 + 200
passaggi del modello per giorno e target sui modelli Alpha158, 8 + 200 sugli Alpha360); permutation
importance per famiglia (7 famiglie × 5 ripetizioni); i pesi interni di MASTER, MATCC e FinFormer. I giorni sono **tutte** le date di ribilanciamento
settimanale (~52 l'anno), per ogni orizzonte, come per gli universi spiegati sul PC di Luca.

Cosa aspettarsi (misure sul PC di Luca, RTX 5090 condivisa da 6 processi, NASDAQ-100, ~100 titoli),
in secondi per giorno spiegato: DiscoverPLF 163, FinFormer 84, FactorVAE 68, HIST 65, MATCC 8, MASTER 4
(MASTER su DJI, ~30 titoli, da solo sulla GPU: 53 giorni in 30 s). Shapley e LIME, su DJI, aggiungono
per giorno: FactorVAE 45 s, DiscoverPLF 13, gli altri 2–3. Il loro costo cresce con il numero di titoli
più lentamente di quello di IG.
Il costo di un giorno cresce più che in proporzione al numero di titoli (ogni titolo è spiegato rispetto
ai propri input, ma il modello vede tutti i titoli insieme): con 500–600 titoli aspettarsi da 4 a 25
volte tanto. La **RAM** di un processo è dominata dal caricamento dei dati dell'universo, come nel
training (picchi del training: sp500 26–63 GiB, sxxp 27–72 GiB). Per questo il passo 6 misura tempi e
RAM prima del lancio.

## Regole

- **Non modificare** il codice dei modelli né quello delle spiegazioni (`code/xaifin/`, `code/scripts/`):
  i risultati devono essere confrontabili con quelli calcolati sul PC di Luca.
- Le spiegazioni scrivono solo in `code/results/xai/`. **Non cancellare nulla** in `code/results/`.
- Se un errore richiede di toccare il codice, **fermati e riporta** il problema (comando, ultime righe
  del log) invece di correggerlo. Sistemare l'ambiente (pacchetti, versione CUDA, percorsi) va bene:
  annota cosa hai fatto in `NOTES_SERVER.md` nella radice.
- Il server può essere condiviso: non occupare più del necessario e lascia margine di RAM.

## Passo 1: ricognizione

```bash
nvidia-smi                      # GPU, memoria, versione driver/CUDA, altri processi
nproc; free -g; df -h .         # core, RAM, disco libero qui
which tmux screen               # serve uno dei due
```

Disco: servono **~400 GB liberi**. Ambiente ~8 GB, `data.zip` 38 GB (cancellabile dopo l'estrazione e il
passo 4), dati estratti ~100 GB in CSV più le copie Parquet, risultati
fino a **~230 GB** a fine lavoro: i tensori delle attribuzioni (un valore per titolo, giorno, giorno passato
e feature), per lo più quelli dei modelli Alpha158 a L = 20, che hanno 60 giorni di input. Se lo spazio
non basta, **fermati e chiedi a Luca** prima di lanciare.

Secondo Luca la cartella del training di ottobre non è più sul server, quindi i dati arrivano con
`data.zip`. Se ne trovi comunque una (`master-thesis/` con `code/data/sp500`), **non estrarre il pacchetto
nuovo sopra quella cartella**: va in una cartella nuova.

## Passo 2: estrazione

Due file:

- `xaifin_xai_bundle.zip`: codice, checkpoint dei 486 run (più uno di DJI per il controllo del passo 5),
  le spiegazioni di riferimento (`xai_reference/`) e questa guida; si estrae in `master-thesis/`;
- `data.zip` (~38 GB compressi): i dati di tutti e cinque gli universi e `constituents`, sotto `data/`;
  si estrae **dentro `master-thesis/code/`**, così che i file finiscano in `master-thesis/code/data/<universo>/`.

```bash
mkdir -p ~/xai && cd ~/xai
unzip /percorso/xaifin_xai_bundle.zip
cd master-thesis/code && unzip /percorso/data.zip && cd ..   # crea code/data/sp500, code/data/sxxp, ...
ls code/data                                                 # constituents dji nasdaq100 sp500 sx5e sxxp
```

Se `ls code/data` mostra una cartella `data` annidata (`code/data/data/...`), sposta il contenuto un
livello sopra. Lo zip è stato confrontato file per file con i dati del PC di Luca: è lo stesso.
NASDAQ-100 ed EURO STOXX 50 qui non servono (li spiega il PC di Luca): per risparmiare disco e il tempo
della conversione in Parquet del passo 3, toglili con `rm -rf code/data/nasdaq100 code/data/sx5e`.

Se invece i dati della cartella del training di ottobre fossero ancora sul server, si possono collegare
invece di estrarre `data.zip` (i file Parquet già creati allora vengono riusati):

```bash
mkdir -p code/data
for d in sp500 sxxp dji constituents; do ln -s /percorso/vecchio/master-thesis/code/data/$d code/data/$d; done
```

## Passo 3: ambiente

```bash
REQS=code/requirements-xai.txt bash code/scripts/remote/setup.sh
```

Installa `uv`, Python 3.12 in `.venv/`, torch CUDA 12.8, i pacchetti delle spiegazioni (captum) e
`xaifin`. Se il driver è troppo vecchio per CUDA 12.8 (colonna "CUDA Version" di `nvidia-smi` < 12.8):
`TORCH_CUDA=cu126 REQS=code/requirements-xai.txt bash code/scripts/remote/setup.sh`.

Alla fine deve stampare `CUDA available: True` e l'elenco delle GPU. Il controllo dei dati segnala come
`MISSING` nasdaq100 e sx5e: è giusto, qui non servono. Poi crea le copie Parquet dei file di feature, che
si caricano molto più in fretta dei CSV: può richiedere decine di minuti e molta RAM. Lo stato del
training stampato in fondo mostra finiti solo i run di sp500 e sxxp (e uno di DJI): è giusto.

Per tutti i comandi successivi: `source .venv/bin/activate`.

## Passo 4: verifica

```bash
python code/scripts/grid.py status --xai --universes sp500 sxxp
```

Deve dire `0/486 runs explained` e **nessun** "runs have no checkpoint". Se mancano checkpoint, il
pacchetto è incompleto: fermati e riporta.

## Passo 5: controllo incrociato con il PC di Luca

```bash
python code/scripts/remote/check_xai.py
```

Spiega qui un run che Luca ha già spiegato (MASTER, DJI, L = 5, seed 42, 2020, tutte le 53 date, tutti i metodi: circa 5 minuti) e confronta i risultati
file per file. Deve finire con `RESULT: the explanations match the reference`. Piccole differenze
numeriche tra GPU diverse sono normali; se compare `DIFF`, **fermati e manda a Luca `xai_check.txt`**
prima di andare avanti.

## Passo 6: prova e misura

Due run veri, uno per ciascun insieme di feature (i risultati restano e non vanno rifatti), misurando
tempo e picco di RAM:

```bash
/usr/bin/time -v python code/scripts/explain.py --models MASTER --universes sp500 --horizons 5 --seeds 0 --years 2020 2>&1 | tail -25
/usr/bin/time -v python code/scripts/explain.py --models HIST --universes sp500 --horizons 5 --seeds 0 --years 2020 2>&1 | tail -25
```

Annota per ciascuno: la riga `... N days, S s` (secondi del run) e `Maximum resident set size` (in kB:
dividere per 1048576 per avere GiB). Durante il secondo, guarda in `nvidia-smi` la memoria GPU del processo.
Scrivi le misure in `NOTES_SERVER.md` e dalle a Luca: servono a stimare la durata totale.

## Passo 7: risorse

`xai.sh` usa queste variabili d'ambiente:

| Variabile | Default | Significato |
|---|---|---|
| `GPUS` | tutte | GPU da usare, es. `GPUS=0` |
| `WORKERS` | 4 | processi contemporanei **per GPU** (al massimo; la RAM decide quanti partono davvero) |
| `THREADS` | 2 | core CPU per processo |
| `MAX_RAM` | 80% della RAM disponibile | GiB che i processi possono usare in tutto |
| `RAM_PER_TASK` | i picchi del training (63–72 GiB) | picco di RAM atteso per processo: usa la misura più alta del passo 6, +15% |
| `CPUS` | nessuno | vincola i processi a questi core, es. `CPUS=0-15` |

Un processo spiega una combinazione (modello, universo, orizzonte, seed), cioè 4–5 run. Parte solo se
la sua RAM attesa ci sta insieme a quella dei processi in corso; se la RAM reale sfora, il più giovane
viene fermato e rimesso in coda (riparte dal run su cui era, il lavoro finito resta). L'ordine è
L = 5 prima di tutto (sono i risultati che Luca vuole per primi), poi L = 1 e 20, i modelli lenti per primi.

Scegli i valori in base al passo 1 e alle misure del passo 6, lasciando margine se il server è condiviso.
Ad esempio, con una H200, ~128 GiB di RAM a disposizione e un picco misurato di 40 GiB:
`GPUS=0 WORKERS=4 MAX_RAM=128 RAM_PER_TASK=46 CPUS=0-15`. Se anche la memoria GPU per processo è alta,
`WORKERS` va scelto perché ci stia nella GPU. **Proponi i valori all'utente prima di lanciare.**

## Passo 8: lancio

```bash
tmux new -s xai 'GPUS=0 WORKERS=4 MAX_RAM=<GiB> RAM_PER_TASK=<GiB> CPUS=<core> bash code/scripts/remote/xai.sh'
```

(con screen: `screen -S xai`, poi lo stesso comando senza `tmux new ...`). Staccarsi: `Ctrl-B D` (tmux)
o `Ctrl-A D` (screen). Ogni processo scrive in `logs/xai/task<N>.log`; il terminale di tmux mostra
partenze, fine e fermate per RAM. Dopo 30 minuti controlla `logs/xai/status.json` (RAM reale per
processo), `free -g` e `nvidia-smi`.

**Cambiare i processi per GPU senza fermare nulla:** `echo 6 > logs/xai/WORKERS` (con meno processi, quelli in
corso finiscono e non vengono sostituiti). Utile dopo i primi controlli, se GPU e RAM hanno margine o sono al limite.

**Fermarsi:** `touch logs/xai/STOP` non avvia processi nuovi e lascia finire quelli in corso;
`tmux kill-session -t xai` ferma subito tutto (al rilancio si riparte dal run interrotto).

**Quando la griglia è finita** (o se resta margine di GPU mentre gira), la valutazione dei metodi sugli
universi grandi: un run per modello × universo × orizzonte, 36 run.

```bash
tmux new -s eval 'source .venv/bin/activate && python code/scripts/evaluate.py --universes sp500 sxxp'
```

## Passo 9: controlli periodici

Dopo che i primi processi a L = 5 sono finiti (per ricalibrare la stima dei tempi), poi una volta al giorno:

```bash
source .venv/bin/activate
python code/scripts/grid.py status --xai --universes sp500 sxxp   # avanzamento per modello, universo, orizzonte
cat logs/xai/status.json                                          # processi in corso, RAM, falliti
grep -l "Traceback" logs/xai/*.log                                # processi falliti
free -g; nvidia-smi; df -h .                                      # risorse, disco
```

Aggiorna in `NOTES_SERVER.md` la stima della fine (run spiegati al giorno). I processi falliti per un
errore Python **non** vengono ritentati automaticamente: vanno riportati (vedi Regole).

## Passo 10: risultati da mandare

Il relatore vuole vedere i risultati a pochi giorni di distanza, quindi **ogni 2–3 giorni** un pacchetto leggero:

```bash
bash code/scripts/remote/pack_xai.sh            # leggero: tutto tranne i tensori delle attribuzioni (1-2 GB a fine lavoro)
```

Il pacchetto leggero contiene tutto ciò che serve alle analisi della tesi. Quello completo
(`FULL=1 bash code/scripts/remote/pack_xai.sh`, fino a ~230 GB) si fa **solo se Luca lo chiede**: per
quelle dimensioni è meglio rsync o un disco fisico, oppure lasciare i tensori sul server finché servono.

Entrambi contengono anche i log, `xai_check.txt` e `NOTES_SERVER.md`. Ogni pacchetto contiene tutto ciò
che è finito fino a quel momento, quindi uno più recente sostituisce i precedenti. Come mandarli, in
ordine di preferenza: **rsync/SSH** se Luca ha accesso al server (copia direttamente `code/results/xai/`);
**Google Drive / OneDrive** con un link condiviso, verificando che il caricamento sia completo (stessa
dimensione in byte del file locale). Dire a Luca a che punto è (`grid.py status --xai ...`) quando lo si manda.

**Non cancellare nulla** dal server finché Luca non conferma di aver ricevuto ed estratto i risultati.

## Problemi comuni

| Sintomo | Causa probabile | Cosa fare |
|---|---|---|
| processi uccisi senza traceback, `dmesg` parla di "Out of memory" | RAM esaurita | fermare, rilanciare con `MAX_RAM` più basso o `RAM_PER_TASK` più alto |
| `CUDA out of memory` nel log | troppi processi per GPU | `WORKERS` più basso |
| tutto molto lento, load average ≫ core dati | CPU satura | `WORKERS` più basso |
| molti "stopping task ... RAM" nel terminale | `RAM_PER_TASK` troppo basso | alzarlo |
| `no CUDA` / `CUDA available: False` | versione torch/driver | rifare il passo 3 con un altro `TORCH_CUDA` |
| `check_xai.py` riporta `DIFF` | ambiente diverso da quello di Luca | non lanciare: mandare `xai_check.txt` a Luca |
| errore Python in un modello o in un metodo | bug | non correggere: riportare comando e log, gli altri processi continuano |
| disco quasi pieno | tensori delle attribuzioni | mandare un pacchetto completo e chiedere a Luca prima di liberare spazio |
