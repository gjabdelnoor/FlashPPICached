#!/usr/bin/env bash
# Forward walk from an arbitrary START index, sharded across the local GPUs.
#
# Replaces the reverse walk. thunder is projected to consume indices 4..111 in
# the next 48 h at its measured 26.7 min/host, so this box starts at 112 and
# walks upward from there: the two never contend for the same host instead of
# converging on an unknown midpoint. predict_host_batch.py still skips any pair
# whose CSV exists, and sync_results.sh merges both boxes' output, so the
# overlap guard survives even if the projection is off.
#
# Usage: host_shard_loop_forward.sh 0 0   -> fan out into TOTAL_SHARDS workers
#        host_shard_loop_forward.sh 2 4   -> I am worker 2 of 4, walk my slice
set -euo pipefail

SHARD_ID=$1
NUM_SHARDS=${2:-4}

PROJ=$HOME/ppi
HOST_DIR=$PROJ/data/proteomes
VIRAL_DIR=$PROJ/data/viral_proteomes
VIRAL_CACHE_DIR=$PROJ/viral_cache
OUT_DIR=$PROJ/output
PY=$PROJ/venv/bin/python

export START=${START:-112}
export BATCH_SIZE=${BATCH_SIZE:-32}
export RESIDUE_DEVICE=${RESIDUE_DEVICE:-cpu}
# Free VRAM held back for activations after the residues are staged. These are
# 11.6 GiB cards: the first attempt here preloaded all 95 viral proteomes on top
# of 3.6 GiB of host residues and every shard died with CUDA OOM inside the
# first pair. predict_host_batch.py now fills the viral cache only up to
# (free VRAM - this), so the same command is correct on a 48 GiB card too.
export VRAM_HEADROOM_GIB=${VRAM_HEADROOM_GIB:-6}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Fan out only on the sentinel 0, never on "1". The old guard was `NUM_SHARDS
# -eq 1`, which is indistinguishable from a legitimate single-worker slice: with
# TOTAL_SHARDS=1 it re-exec'd `$0 0 1`, hit the same guard, and recursed until
# the box had 157 copies of this script fighting for the CPU. A sentinel that
# cannot collide with a real shard count removes the trap entirely.
if [ "$NUM_SHARDS" -eq 0 ]; then
    TOTAL=${TOTAL_SHARDS:-$NGPU}
    for s in $(seq 0 $((TOTAL - 1))); do
        CUDA_VISIBLE_DEVICES=$((s % NGPU)) "$0" "$s" "$TOTAL" &
    done
    wait
    exit
fi

mkdir -p "$OUT_DIR"

mapfile -t HOSTS < <(ls "$HOST_DIR"/*.faa | sort)
NH=${#HOSTS[@]}

idx=$((START + SHARD_ID))
while [ "$idx" -lt "$NH" ]; do
    host="${HOSTS[$idx]}"
    echo "== [fshard $SHARD_ID gpu ${CUDA_VISIBLE_DEVICES:-all}] idx $idx $(basename "$host")"
    "$PY" "$PROJ/repo/predict_host_batch.py" \
        --host_fasta "$host" \
        --viral_dir "$VIRAL_DIR" \
        --viral_cache_dir "$VIRAL_CACHE_DIR" \
        --output_dir "$OUT_DIR" \
        --model_name "$PROJ/models/flashppi" \
        --batch_size "$BATCH_SIZE" \
        --residue_device "$RESIDUE_DEVICE" \
        --vram_headroom_gib "$VRAM_HEADROOM_GIB" \
        >> "$OUT_DIR/fshard${SHARD_ID}.log" 2>&1
    idx=$((idx + NUM_SHARDS))
done

echo "forward shard $SHARD_ID done" >> "$OUT_DIR/fshard${SHARD_ID}.log"
