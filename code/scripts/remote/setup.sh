#!/usr/bin/env bash
# One-time setup on a Linux GPU server: Python 3.12 environment with CUDA torch, then a check of
# the GPUs and the data. Run from anywhere:  bash code/scripts/remote/setup.sh
# Other CUDA build (old driver):  TORCH_CUDA=cu126 bash code/scripts/remote/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repository root
TORCH_CUDA=${TORCH_CUDA:-cu128}

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/.uv-cache}

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
uv pip install -r code/requirements-train.txt
uv pip install -e code --no-deps

python - <<'EOF'
import torch
from xaifin.config import DATA_ROOT, UNIVERSES
from xaifin.data.loading import alpha_path

print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
missing = [str(alpha_path(u, k).with_suffix(".parquet").relative_to(DATA_ROOT)) for u in UNIVERSES
           for k in ("alpha158", "alpha360")
           if not alpha_path(u, k).with_suffix(".parquet").exists() and not alpha_path(u, k).exists()]
if not (DATA_ROOT / "constituents").is_dir():
    missing.append("constituents/")
print(f"data in {DATA_ROOT}: " + (f"MISSING {missing}" if missing else "all universes present"))
EOF
python code/scripts/convert_data.py   # Parquet copies of the feature CSVs, if any are still CSV only
python code/scripts/grid.py status
