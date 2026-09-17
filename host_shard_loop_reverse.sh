#!/usr/bin/env bash
# Reverse-order companion to host_shard_loop.sh.
#
# thunder walks the sorted host list upward from index 0; this walks it
# downward from the last index. The two converge in the middle, and whichever
# reaches a host second skips it -- predict_host_batch.py treats any pair whose
# CSV already exists as done. sync_results.sh keeps both machines' output
# directories merged, so "already done" includes the other machine's work.
#
# Usage: host_shard_loop_reverse.sh <shard_id> <num_shards>
#        host_shard_loop_reverse.sh 0 1   -> fans out into NUM_SHARDS workers
set -euo pipefail

SHARD_ID=$1
NUM_SHARDS=${2:-4}

PROJ=$HOME/ppi
HOST_DIR=$PROJ/data/proteomes
VIRAL_DIR=$PROJ/data/viral_proteomes
VIRAL_CACHE_DIR=$PROJ/viral_cache
OUT_DIR=$PROJ/output
PY=$PROJ/venv/bin/python

# See host_shard_loop.sh: bucketing hands the allocator a new shape every batch,
# and without this a shard drifts to ~3x its working set.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "$NUM_SHARDS" -eq 1 ]; then
    TOTAL=${TOTAL_SHARDS:-4}
    for s in $(seq 0 $((TOTAL - 1))); do
        # Spread shards round-robin over the visible GPUs.
        CUDA_VISIBLE_DEVICES=$((s % NGPU)) "$0" "$s" "$TOTAL" &
    done
    wait
    exit
fi

mkdir -p "$OUT_DIR"

mapfile -t HOSTS < <(ls "$HOST_DIR"/*.faa | sort)
NH=${#HOSTS[@]}

# Descend from the end so the index space is consumed from the opposite side.
idx=$((NH - 1 - SHARD_ID))
while [ "$idx" -ge 0 ]; do
    host="${HOSTS[$idx]}"
    echo "== [rshard $SHARD_ID gpu ${CUDA_VISIBLE_DEVICES:-all}] idx $idx $(basename "$host")"
    "$PY" "$PROJ/repo/predict_host_batch.py" \
        --host_fasta "$host" \
        --viral_dir "$VIRAL_DIR" \
        --viral_cache_dir "$VIRAL_CACHE_DIR" \
        --output_dir "$OUT_DIR" \
        --model_name "$PROJ/models/flashppi" \
        --batch_size 32 \
        >> "$OUT_DIR/rshard${SHARD_ID}.log" 2>&1
    idx=$((idx - NUM_SHARDS))
done

echo "reverse shard $SHARD_ID done" >> "$OUT_DIR/rshard${SHARD_ID}.log"
