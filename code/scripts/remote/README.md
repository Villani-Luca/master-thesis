# Training su un server GPU

Istruzioni per far girare il training della tesi su un server Linux con GPU NVIDIA (es. 2× H200).
La guida passo passo, pensata anche per Claude Code, è `START_HERE.md` nella radice del pacchetto
(qui: [START_HERE.md](START_HERE.md)); questo file è il riassunto.

## Cosa fa

Allena 6 modelli di previsione azionaria su 5 universi, 3 orizzonti e 3 seed, con finestre mobili
(anni di test 2020–2024): 1242 run in tutto, alcuni già fatti. I modelli sono piccoli (pochi MB,
≤ 2,3 GiB di memoria GPU a run), quindi la GPU si sfrutta facendo girare **molti run in parallelo**
sulla stessa GPU: il limite pratico è la RAM (fino a ~72 GiB per un run su sxxp), non la GPU.

## Requisiti

- Linux, driver NVIDIA ≥ 570 (per torch CUDA 12.8; con un driver più vecchio vedi sotto)
- ~60 GB liberi su disco (dati ~45 GB, ambiente Python ~8 GB, risultati fino a ~5 GB)
- `tmux` o `screen` per lasciare girare il training dopo il logout
- accesso a internet per l'installazione (scarica Python 3.12 e torch)

## Passi

```bash
unzip xaifin_bundle.zip && cd master-thesis
bash code/scripts/remote/setup.sh
```

Alla fine `setup.sh` stampa le GPU viste da torch, se i dati ci sono tutti
(`code/data/<universo>/` e `code/data/constituents/`) e lo stato della griglia. Se il driver non
supporta CUDA 12.8: `TORCH_CUDA=cu126 bash code/scripts/remote/setup.sh`.

Poi:

```bash
tmux new -s xaifin 'GPUS=0 MAX_RAM=128 CPUS=0-15 bash code/scripts/remote/train.sh'
```

- `Ctrl-B` poi `D` per staccarsi; `tmux attach -t xaifin` per tornare a vedere.
- `GPUS` = GPU da usare (default tutte), `WORKERS` = run al massimo per GPU (default 8),
  `THREADS` = core per run (default 2), `MAX_RAM` = GiB per tutti i run (default 80% della RAM
  disponibile), `CPUS` = core a cui vincolarli (default nessun vincolo). Un run parte solo se la sua
  RAM stimata ci sta; se la RAM reale sfora, il più giovane viene fermato e rimesso in coda.
- Ogni run scrive il suo log in `logs/grid/task<N>.log`; lo stato dal vivo è in `logs/grid/status.json`.
- Lo stato della griglia: `source .venv/bin/activate && python code/scripts/grid.py status`.
- `touch logs/grid/STOP`: non avvia più run nuovi, lascia finire quelli in corso.

## Interruzioni

Si può fermare in qualsiasi momento (Ctrl-C, riavvio, kill): basta rilanciare lo stesso comando
`tmux new ...`. I run finiti vengono saltati e quelli a metà ripartono dall'ultima epoca salvata.

## Risultati

Quando ha finito (o anche a metà):

```bash
bash code/scripts/remote/pack_results.sh
```

crea `xaifin_results_<host>_<data>.tar.gz` con checkpoint, predizioni, metriche e log: è il file
da rimandare indietro.

## Se qualcosa va storto

- **Un run fallisce**: il riepilogo finale dice quanti; il motivo è in fondo al suo
  `logs/grid/task<N>.log`. Rilanciare `train.sh` ritenta solo quelli non finiti.
- **Memoria GPU esaurita** (`CUDA out of memory` nei log): rilanciare con meno run per GPU,
  es. `WORKERS=4`.
- **CPU satura / tutto lento**: abbassare `WORKERS`.
- **RAM esaurita** (processi uccisi senza errore Python, `dmesg` parla di OOM): ogni run carica in
  memoria i dati del suo universo, e S&P 500 / STOXX 600 pesano di più. Abbassare `MAX_RAM`.
