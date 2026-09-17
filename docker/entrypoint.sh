#!/usr/bin/env bash
# Container entrypoint for the viral-host PPI sweep.
#
# Subcommands: sweep (default) | warm-cache | pair | version | <any command>
set -euo pipefail

: "${HOST_DIR:=/data/proteomes}"
: "${VIRAL_DIR:=/data/viral_proteomes}"
: "${VIRAL_CACHE_DIR:=/cache}"
: "${OUT_DIR:=/output}"
: "${FLASHPPI_MODEL:=/opt/models/flashppi}"
: "${NUM_SHARDS:=3}"
: "${BATCH_SIZE:=32}"

# Length bucketing hands the allocator a different shape every batch, so freed
# blocks rarely fit the next request and get hoarded rather than reused. With
# spare VRAM nothing forces a reclaim and a shard drifts to ~23 GB on a working
# set under 8 GB. Over 300 realistic batches this cut peak reserved from
# 11.39 GiB to 6.55 GiB and removed the forced reclaims.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd /opt/flashppi

warm_cache() {
    mkdir -p "$VIRAL_CACHE_DIR"
    python warm_viral_cache.py \
        --viral_dir "$VIRAL_DIR" \
        --cache_dir "$VIRAL_CACHE_DIR" \
        --model_name "$FLASHPPI_MODEL" \
        --batch_size "$BATCH_SIZE"
}

# One shard walks every NUM_SHARDS-th host. predict_host_batch.py skips any
# (host, viral) pair whose CSV already exists, so a killed shard resumes where
# it stopped rather than redoing the proteome.
run_shard() {
    local shard_id=$1
    mapfile -t hosts < <(ls "$HOST_DIR"/*.faa | sort)
    local idx=$shard_id
    while [ "$idx" -lt "${#hosts[@]}" ]; do
        python predict_host_batch.py \
            --host_fasta "${hosts[$idx]}" \
            --viral_dir "$VIRAL_DIR" \
            --viral_cache_dir "$VIRAL_CACHE_DIR" \
            --output_dir "$OUT_DIR" \
            --model_name "$FLASHPPI_MODEL" \
            --batch_size "$BATCH_SIZE" \
            >> "$OUT_DIR/hostshard${shard_id}.log" 2>&1
        idx=$((idx + NUM_SHARDS))
    done
    echo "host shard $shard_id done" >> "$OUT_DIR/hostshard${shard_id}.log"
}

case "${1:-sweep}" in
    sweep)
        mkdir -p "$OUT_DIR"
        # Residue embeddings live in system RAM, not VRAM (cache_utils ends each
        # tensor with .cpu().half()), so a shard costs 2.78 GiB of weights plus
        # an activation peak set by BATCH_SIZE -- about 8 GiB at 32. Budget
        # ~9 GiB of VRAM and ~5 GiB of RAM per shard.
        [ -n "$(ls -A "$VIRAL_CACHE_DIR" 2>/dev/null)" ] || warm_cache
        for s in $(seq 0 $((NUM_SHARDS - 1))); do
            run_shard "$s" &
        done
        wait
        ;;
    warm-cache)
        warm_cache
        ;;
    pair)
        shift
        python predict_cross_proteome.py "$@"
        ;;
    version)
        python - <<'PY'
import faiss, torch, transformers
print("python      ", __import__("sys").version.split()[0])
print("torch       ", torch.__version__)
print("cuda build  ", torch.version.cuda)
print("cuda avail  ", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("faiss       ", faiss.__version__)
PY
        ;;
    *)
        exec "$@"
        ;;
esac
