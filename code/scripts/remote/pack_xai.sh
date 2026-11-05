#!/usr/bin/env bash
# The explanations of the big universes, in one archive to send back, with the logs and the cross-check.
#
#     bash code/scripts/remote/pack_xai.sh           # light: all but the attribution tensors (attributions_*.npz)
#     FULL=1 bash code/scripts/remote/pack_xai.sh    # everything, attribution tensors included (up to ~230 GB)
#
# The light archive holds what the analysis reads (global importance, permutation, intrinsic weights,
# parameters): it is the one to send every few days. The full one is for the end.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repository root
UNIVERSES=${UNIVERSES:-sp500 sxxp}
paths=()
for u in $UNIVERSES; do
    for d in code/results/xai/*/*/"$u"; do [ -d "$d" ] && paths+=("$d"); done
done
[ ${#paths[@]} -gt 0 ] || { echo "nothing explained yet for: $UNIVERSES"; exit 1; }
for f in code/results/xai/explain.log logs/xai xai_check.txt NOTES_SERVER.md; do [ -e "$f" ] && paths+=("$f"); done
# logs and notes go to logs/xai_server/ in the archive, so that extracting it at home overwrites no local log
moved=(--transform='s,^code/results/xai/explain\.log,logs/xai_server/explain.log,'
       --transform='s,^logs/xai\($\|/\),logs/xai_server\1,'
       --transform='s,^xai_check\.txt,logs/xai_server/xai_check.txt,'
       --transform='s,^NOTES_SERVER\.md,logs/xai_server/NOTES_SERVER.md,')
if [ "${FULL:-0}" = 1 ]; then
    out="xaifin_xai_full_$(hostname)_$(date +%Y%m%d_%H%M).tar"   # npz and parquet are compressed already
    tar "${moved[@]}" -cf "$out" "${paths[@]}"
else
    out="xaifin_xai_light_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz"
    tar "${moved[@]}" --exclude='attributions_*.npz' -czf "$out" "${paths[@]}"
fi
echo "$out ($(du -h "$out" | cut -f1))"
