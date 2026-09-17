#!/usr/bin/env bash
# Parallel worker: processes every Nth HOST (not pair). Each host is
# encoded exactly once and reused in-memory for all 95 viral pairings
# inside predict_host_batch.py. Viral embeddings come from the on-disk
# cache built by warm_viral_cache.py, not re-encoded per pair.
set -euo pipefail

SHARD_ID=$1
NUM_SHARDS=$2

# Length bucketing gives every batch a different shape, so freed blocks rarely
# fit the next request and the caching allocator hoards them instead of reusing
# them. Left alone on a 48 GB card nothing ever forces a reclaim and each shard
# drifts to ~23 GB while its working set stays under 8 GB. Measured over 300
# realistic batches: peak reserved 11.39 GiB by default vs 6.55 GiB here, and
# the repeated forced reclaims disappear too.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Three shards, not two: with the allocator bounded, VRAM stops being the limit
# and GPU utilization becomes it. Two shards left the card 28% idle waiting on
# per-pair CPU work (torch.load of the viral cache, FAISS, device transfers).
# Not four -- there are only 6 cores, and that CPU work is what fills the gaps.
if [ "$NUM_SHARDS" -eq 1 ]; then
    "$0" 0 3 &
    "$0" 1 3 &
    "$0" 2 3 &
    wait
    exit
fi

PROJ=/home/ubuntu/projects/ppi/flashppi-proteome
HOST_DIR=/home/ubuntu/projects/ppi/data/proteomes
VIRAL_DIR=/home/ubuntu/projects/ppi/data/viral_proteomes
VIRAL_CACHE_DIR=/home/ubuntu/projects/ppi/viral_cache
OUT_DIR=/home/ubuntu/projects/ppi/output

source "$PROJ/models/ppi/bin/activate"
mkdir -p "$OUT_DIR"

mapfile -t HOSTS < <(ls "$HOST_DIR"/*.faa | sort)
NH=${#HOSTS[@]}

idx=$SHARD_ID
while [ "$idx" -lt "$NH" ]; do
    host="${HOSTS[$idx]}"
    echo "== [shard $SHARD_ID] host $(basename "$host")"
    python "$PROJ/repo/predict_host_batch.py" \
        --host_fasta "$host" \
        --viral_dir "$VIRAL_DIR" \
        --viral_cache_dir "$VIRAL_CACHE_DIR" \
        --output_dir "$OUT_DIR" \
        --model_name "$PROJ/models/flashppi" \
        --batch_size 32 \
        >> "$OUT_DIR/hostshard${SHARD_ID}.log" 2>&1
    idx=$((idx + NUM_SHARDS))
done

echo "host shard $SHARD_ID done" >> "$OUT_DIR/hostshard${SHARD_ID}.log"
