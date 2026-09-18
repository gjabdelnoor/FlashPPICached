#!/usr/bin/env bash
# Kill by tmux session name, never by pkill on the script name -- `pkill -f
# predict_host_batch` matches the ssh command line that carries it and kills the
# relaunch before it runs. That has cost idle GPU time twice now.
set -euo pipefail
N=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
# Guard the shard count before it can bite. Passing 1 here used to fan out to a
# single worker which fanned out again -- 157 concurrent copies of the loop
# script on a 6-vCPU box. The loop now keys fan-out off the sentinel 0, so this
# asserts the count is sane rather than relying on the caller to know that.
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "restart_forward.sh: shard count must be a positive integer, got '$N'" >&2
    exit 2
fi
tmux kill-session -t rppi 2>/dev/null || true
tmux kill-session -t fppi 2>/dev/null || true
sleep 3
mkdir -p "$HOME/ppi/output"
rm -f "$HOME"/ppi/output/fshard*.log
tmux new-session -d -s fppi \
  "TOTAL_SHARDS=$N START=${START:-112} BATCH_SIZE=${BATCH_SIZE:-32} RESIDUE_DEVICE=${RESIDUE_DEVICE:-cuda} VRAM_HEADROOM_GIB=${VRAM_HEADROOM_GIB:-6} bash $HOME/ppi/host_shard_loop_forward.sh 0 0 > $HOME/ppi/output/launcher.log 2>&1"
echo "launched $N forward shards from idx ${START:-112}"
