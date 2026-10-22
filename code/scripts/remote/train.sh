#!/usr/bin/env bash
# Train every unfinished run of the grid on the GPUs of this server. Resumable: run it again after
# any stop and it continues where it was (finished runs skipped, others resume from their last
# epoch). Start it inside tmux or screen so it survives the logout:
#
#     tmux new -s xaifin 'bash code/scripts/remote/train.sh'
#
# GPUS    = GPU ids to use (default: all visible), e.g. GPUS=0 for one GPU only
# WORKERS = runs at the same time on each GPU (default 8: the runs are CPU-bound, one alone uses a
#           GPU at 4-34%, so several share it; RAM decides how many really start, see grid.py)
# THREADS = CPU threads per run (default 2: same results and speed as 6, measured on an H200)
# MAX_RAM = GiB of RAM all runs may use together (default: 80% of the RAM available at the start)
# CPUS    = CPU ids to pin all runs to, e.g. CPUS=0-15 to keep to 16 cores (default: no pinning)
# Extra arguments go to grid.py, e.g.  bash code/scripts/remote/train.sh --universes sp500 sxxp
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repository root
source .venv/bin/activate

GPUS=${GPUS:-$(python -c "import torch; print(' '.join(map(str, range(torch.cuda.device_count()))))")}
[ -n "$GPUS" ] || { echo "no GPU visible to torch"; exit 1; }
THREADS=${THREADS:-2}
WORKERS=${WORKERS:-8}
EXTRA=()
[ -n "${MAX_RAM:-}" ] && EXTRA+=(--max-ram "$MAX_RAM")
[ -n "${CPUS:-}" ] && EXTRA+=(--cpus "$CPUS")

echo "GPUs: $GPUS, up to $WORKERS runs per GPU, $THREADS CPU threads each ${EXTRA[*]:-}"
python code/scripts/grid.py run --gpus $GPUS --workers "$WORKERS" --threads "$THREADS" --device cuda \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@"
python code/scripts/grid.py status
