#!/usr/bin/env bash
# Everything trained so far, in one archive to send back: checkpoints, predictions, metrics, logs.
# Half-trained runs (resume.pt) are left out.  bash code/scripts/remote/pack_results.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repository root
out="xaifin_results_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz"
paths=(code/results/Regression)
[ -d logs ] && paths+=(logs)
tar --exclude='resume.pt' --exclude='resume.tmp' -czf "$out" "${paths[@]}"
echo "$out ($(du -h "$out" | cut -f1))"
