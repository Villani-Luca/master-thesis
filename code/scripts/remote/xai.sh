#!/usr/bin/env bash
# Explain every trained run of the big universes that is not explained yet, on the GPUs of this server
# (grid.py run --xai: explain.py, every method, both targets). Resumable: run it again after any stop
# and it continues where it was (explained runs skipped). Start it inside tmux or screen:
#
#     tmux new -s xai 'bash code/scripts/remote/xai.sh'
#
# GPUS         = GPU ids to use (default: all visible), e.g. GPUS=0
# WORKERS      = tasks at the same time on each GPU (default 4; RAM decides how many really start)
# THREADS      = CPU threads per task (default 2)
# MAX_RAM      = GiB of RAM all tasks may use together (default: 80% of the RAM available at the start)
# RAM_PER_TASK = expected peak GiB of one task (default: the training peaks of grid.py, 63-72 GiB on the big
#                universes); set it to what the trial of START_HERE.md step 4 measured, plus a margin
# CPUS         = CPU ids to pin all tasks to, e.g. CPUS=0-15 (default: no pinning)
# UNIVERSES    = universes to explain (default: sp500 sxxp)
# Extra arguments go to grid.py, e.g.  bash code/scripts/remote/xai.sh --horizons 5
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repository root
source .venv/bin/activate

GPUS=${GPUS:-$(python -c "import torch; print(' '.join(map(str, range(torch.cuda.device_count()))))")}
[ -n "$GPUS" ] || { echo "no GPU visible to torch"; exit 1; }
THREADS=${THREADS:-2}
WORKERS=${WORKERS:-4}
UNIVERSES=${UNIVERSES:-sp500 sxxp}
EXTRA=()
[ -n "${MAX_RAM:-}" ] && EXTRA+=(--max-ram "$MAX_RAM")
[ -n "${RAM_PER_TASK:-}" ] && EXTRA+=(--ram-per-task "$RAM_PER_TASK")
[ -n "${CPUS:-}" ] && EXTRA+=(--cpus "$CPUS")

echo "GPUs: $GPUS, up to $WORKERS tasks per GPU, $THREADS CPU threads each, universes: $UNIVERSES ${EXTRA[*]:-}"
python code/scripts/grid.py run --xai --universes $UNIVERSES --gpus $GPUS --workers "$WORKERS" --threads "$THREADS" \
    --device cuda ${EXTRA[@]+"${EXTRA[@]}"} "$@"
python code/scripts/grid.py status --xai --universes $UNIVERSES
