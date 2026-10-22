# START HERE: avviare il training su questo server

Guida operativa per Claude Code (o per chi lo fa a mano). Chiedi a Claude:
**"Leggi START_HERE.md e seguilo passo per passo, chiedendomi conferma prima del lancio finale."**

## Contesto

Tesi magistrale (Luca Villani, UNIMORE) su modelli di previsione azionaria. Va allenata una griglia di
**1242 run**: 6 modelli (MASTER, MATCC, FactorVAE, HIST, DiscoverPLF, FinFormer) × 5 universi
(dji, nasdaq100, sp500, sx5e, sxxp) × 3 orizzonti (L = 1, 5, 20) × 3 seed × anni di test
2020–2024. **117 run sono già fatti** e vengono saltati automaticamente.

I modelli sono piccoli: un run da solo usa una H200 al 4–34% e ≤ 2,3 GiB di VRAM, perché è limitato
dalla latenza CPU/Python. Il guadagno viene dal farne girare **molti in parallelo sulla stessa GPU**;
il limite vero è la **RAM**.

Misure su una H200 NVL (ottobre 2026), tutte con risultati bit-identici al run singolo:

- tempo per run (un anno): FactorVAE ~2,5 h, DiscoverPLF 0,5–0,9 h, MATCC 0,6–0,8 h, HIST ~0,4 h,
  FinFormer 0,14–0,8 h, MASTER ~0,1 h;
- picco di RAM per run: dji 3–6 GiB, sx5e 4–10, nasdaq100 7–17, sp500 26–63, sxxp 27–72
  (FactorVAE ~30 sui due grandi);
- `THREADS=2` e `THREADS=6` danno pesi identici alla stessa velocità; con 8 run sulla stessa GPU
  la resa è circa 3× quella di un run alla volta.

Lavoro residuo: ~39 giorni-GPU in serie; con una H200, 8 run per GPU e ~128 GiB di RAM ≈ **13 giorni**
(12–17 a seconda delle epoche).

## Regole

- **Non modificare** modelli, iperparametri, seed, protocollo o selezione dell'epoca: i risultati
  devono restare confrontabili con quelli già fatti e con il paper FinBench.
- **Non usare `--force`** e non cancellare nulla in `code/results/`: lì ci sono i run finiti
  (`metrics.json`) e quelli a metà (`resume.pt`, da cui il training riprende).
- Se un errore richiede di toccare il codice, **fermati e riporta** il problema (comando, ultime
  righe del log) invece di correggerlo in silenzio. Sistemare l'ambiente (pacchetti, versione CUDA,
  percorsi) va bene: annota cosa hai fatto in `NOTES_SERVER.md` nella radice.
- Il server può essere condiviso: non occupare più del necessario e lascia margine di RAM.

## Passo 1: ricognizione

```bash
nvidia-smi                      # GPU, memoria, versione driver/CUDA, altri processi
nproc; free -g; df -h .         # core, RAM, disco libero qui
which tmux screen               # serve uno dei due
```

Servono: GPU NVIDIA visibili, **~60 GB liberi** su questo disco (ambiente ~8 GB, risultati fino a
~5 GB, il resto sono già i dati), `tmux` o `screen`. Se lo zip del pacchetto è ancora qui, si può
cancellare dopo aver verificato che `code/data/` sia completo (passo 3).

## Passo 2: ambiente

```bash
bash code/scripts/remote/setup.sh
```

Installa `uv` (in `~/.local/bin`), Python 3.12 in `.venv/`, torch CUDA 12.8 e il pacchetto
`xaifin`. Se il driver è troppo vecchio per CUDA 12.8 (driver < 570, colonna "CUDA Version" di
`nvidia-smi` < 12.8): `TORCH_CUDA=cu126 bash code/scripts/remote/setup.sh` (o `cu124`).

Alla fine deve stampare `CUDA available: True`, l'elenco delle GPU e
`all universes present`. Per tutti i comandi successivi: `source .venv/bin/activate`.

## Passo 3: verifica

```bash
source .venv/bin/activate
python code/scripts/grid.py status
```

Deve dire `117/1242 runs finished`. Se dice 0, i risultati già fatti non sono stati estratti
in `code/results/Regression/`.

## Passo 4: prova breve e misura della memoria

Un run vero e corto (MASTER, DJI, 2020, L = 1: pochi minuti), per vedere che tutto funziona:

```bash
python code/scripts/train_rolling.py --models MASTER --universes dji --years 2020 --seeds 0 --sl 5 --pl 1
```

Deve finire con `kept epoch ... test IC ...` e creare
`code/results/Regression/MASTER/dji/sl5_pl1/seed0/y2020/metrics.json`.

## Passo 5: risorse da dare al training

`train.sh` usa questi parametri (variabili d'ambiente), già tarati sulle misure qui sopra:

| Variabile | Default | Significato |
|---|---|---|
| `GPUS` | tutte | GPU da usare, es. `GPUS=0` per una sola H200 |
| `WORKERS` | 8 | run contemporanei **per GPU** (al massimo; la RAM decide quanti partono davvero) |
| `THREADS` | 2 | core CPU per run |
| `MAX_RAM` | 80% della RAM disponibile | GiB che i run possono usare in tutto |
| `CPUS` | nessuno | vincola tutti i run a questi core, es. `CPUS=0-15` |

Come funziona: un run parte solo se il suo picco di RAM stimato (tabella del contesto, +15%) sta
nel 90% di `MAX_RAM` insieme ai run in corso; se la RAM reale supera il 97% di `MAX_RAM`, il run
più giovane viene fermato e rimesso in coda (riprende dall'ultima epoca, risultato identico).
Con 128 GiB sta al più un HIST/DiscoverPLF su sxxp alla volta, più run piccoli.

Scegli i valori in base al server (`nvidia-smi`, `nproc`, `free -g` del passo 1), lasciando margine
se è condiviso: ad esempio per una H200 con ~16 core e ~128 GiB a disposizione
`GPUS=0 WORKERS=8 THREADS=2 MAX_RAM=128 CPUS=0-15` (con più core sul nodo NUMA della GPU si possono
scegliere quelli, ma il guadagno misurato è < 3%). **Proponi i valori all'utente prima di lanciare.**

## Passo 6: lancio

```bash
tmux new -s xaifin 'GPUS=0 WORKERS=8 MAX_RAM=<GiB> CPUS=<core> bash code/scripts/remote/train.sh'
```

Servono tutti gli orizzonti (L = 1, 5, 20): lanciare la griglia intera in una volta, così lo
scheduler affianca run grandi e piccoli e la RAM resta piena.

(con screen: `screen -S xaifin`, poi lo stesso comando senza `tmux new ...`). Staccarsi:
`Ctrl-B D` (tmux) o `Ctrl-A D` (screen). I run più lunghi partono per primi. Ogni run scrive in
`logs/grid/task<N>.log`; il terminale di tmux mostra start/fine di ogni run e gli stop per RAM.

Dopo 30 minuti controlla `logs/grid/status.json` (RAM reale per run), `free -g` e `nvidia-smi`.

**Fermarsi:** `touch logs/grid/STOP` non avvia più run nuovi e lascia finire quelli in corso
(stop morbido); `tmux kill-session -t xaifin` ferma subito tutto (i run riprendono dall'ultima
epoca al rilancio).

## Passo 7: controlli periodici

Dopo ~6 ore (per ricalibrare la stima dei tempi sui primi run finiti), poi una volta al giorno:

```bash
source .venv/bin/activate
python code/scripts/grid.py status          # avanzamento
cat logs/grid/status.json                   # run in corso, RAM, falliti
grep -l "Traceback" logs/grid/*.log         # run falliti
free -g; nvidia-smi                         # risorse
```

I run falliti per un errore Python **non** vengono ritentati automaticamente: vanno riportati (vedi
Regole).

## Passo 8: interruzioni e ripresa

Si può fermare tutto in qualsiasi momento (`touch logs/grid/STOP`, Ctrl-C nella sessione tmux,
`tmux kill-session -t xaifin`, riavvio). Per ripartire basta rilanciare il comando del passo 6, anche
con altri `WORKERS`/`MAX_RAM`: i run finiti vengono saltati, quelli a metà riprendono dall'ultima epoca salvata con
risultati identici a un run mai interrotto.

## Problemi comuni

| Sintomo | Causa probabile | Cosa fare |
|---|---|---|
| processi uccisi senza traceback, `dmesg` parla di "Out of memory" | RAM esaurita | fermare, rilanciare con `MAX_RAM` più basso |
| `CUDA out of memory` nel log | troppi run per GPU | `WORKERS` più basso |
| tutto molto lento, load average ≫ core dati | CPU satura | `WORKERS` più basso |
| molti "stopping task ... RAM" nel terminale | stime di RAM basse per questo server | `MAX_RAM` più basso |
| `no CUDA` / `CUDA available: False` | versione torch/driver | rifare il passo 2 con un altro `TORCH_CUDA` |
| errore Python dentro un modello | bug | non correggere: riportare comando e log, gli altri run continuano |

Un run fallito non blocca gli altri; a fine giro il riepilogo dice quali sono falliti (anche in
`logs/grid/status.json`). Una volta riportato e risolto il problema, rilanciare `train.sh` ritenta
solo quelli non finiti.

## Passo 9: risultati

Quando `grid.py status` dice 1242/1242 (o anche prima, per un invio parziale):

```bash
bash code/scripts/remote/pack_results.sh
```

Crea `xaifin_results_<host>_<data>.tar.gz` (checkpoint, predizioni, metriche, log; i run a metà
restano fuori): è il file da mandare a Luca, insieme a `NOTES_SERVER.md` se esiste.

**Come mandarlo.** Il file pesa qualche GB (stima 4–8 GB a griglia completa), troppo per una mail
o per WeTransfer gratuito (3 GB). In ordine di preferenza:

1. **rsync/SSH**, se Luca ha un accesso al server: non serve il `.tar.gz`, copia lui
   direttamente `code/results/Regression/` e `logs/` (incrementale, riprende se cade).
2. **Google Drive / OneDrive**: caricare il `.tar.gz` in una cartella condivisa e mandare il link.
   Verificare che il caricamento sia completo (stessa dimensione in byte del file locale).

**Invii parziali.** Si può mandare un pacchetto anche a metà lavoro: ogni `pack_results.sh`
contiene tutto quello che è finito fino a quel momento, quindi un pacchetto più recente sostituisce
i precedenti. Dire a Luca a che punto è (`grid.py status`) quando lo si manda.

**Non cancellare nulla** dal server (`code/results/`, `logs/`, il `.tar.gz`) finché Luca non
conferma di aver ricevuto ed estratto i risultati: quella è l'unica copia dei checkpoint.
