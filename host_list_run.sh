#!/usr/bin/env bash
# Process an explicit LIST of hosts, rather than walking the index space.
#
# Why this exists: the three A6000s each walk forward from a fixed start and
# stop wherever their rental budget runs out. That leaves three seams --
# ranges past each box's stop point that nobody will ever reach. Those seams
# are not contiguous, so an index walk is the wrong tool twice over: it would
# spend 20 minutes skipping through the thousands of pairs already done, and
# worse, it would walk straight into the range a box is still working on and
# duplicate it.
#
# So take the list as data. Hosts already complete are skipped inside
# predict_host_batch.py, which makes the list safe to re-run: restart it after
# an interruption and the finished hosts cost one model load each.
#
# Usage: host_list_run.sh <host-list-file>
#        the file holds one host basename per line, e.g. "ECOR-63proteome"
set -euo pipefail

LIST=${1:?usage: host_list_run.sh <host-list-file>}
[ -s "$LIST" ] || { echo "host list $LIST is empty" >&2; exit 2; }

PROJ=${PROJ:-$HOME/projects/ppi}
HOST_DIR=$PROJ/data/proteomes
VIRAL_DIR=$PROJ/data/viral_proteomes
VIRAL_CACHE_DIR=$PROJ/viral_cache
OUT_DIR=$PROJ/output
PY=${PY:-$PROJ/venv/bin/python}

# 12 GiB here versus 48 GiB on the A6000s. predict_host_batch.py budgets both
# VRAM phases against free memory, so it fills what fits and spills the rest to
# host RAM -- correct on either card, just slower here. The contact loop's
# 2-layer transformer is the phase that actually needs room, so leave it more.
export RESIDUE_DEVICE=${RESIDUE_DEVICE:-cuda}
export BATCH_SIZE=${BATCH_SIZE:-16}
export VRAM_HEADROOM_GIB=${VRAM_HEADROOM_GIB:-5}
export ENCODE_HEADROOM_GIB=${ENCODE_HEADROOM_GIB:-1.5}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT_DIR"

n=0; total=$(grep -c . "$LIST")
while read -r name; do
    name=$(echo "$name" | tr -d '[:space:]')
    [ -z "$name" ] && continue
    n=$((n + 1))
    host="$HOST_DIR/${name%.faa}.faa"
    if [ ! -f "$host" ]; then
        echo "== [$n/$total] $name SKIP (no such fasta)"
        continue
    fi
    echo "== [$n/$total] $name $(date -Is)"
    "$PY" "$PROJ/repo/predict_host_batch.py" \
        --host_fasta "$host" \
        --viral_dir "$VIRAL_DIR" \
        --viral_cache_dir "$VIRAL_CACHE_DIR" \
        --output_dir "$OUT_DIR" \
        --model_name "$PROJ/flashppi-proteome/models/flashppi" \
        --batch_size "$BATCH_SIZE" \
        --residue_device "$RESIDUE_DEVICE" \
        --vram_headroom_gib "$VRAM_HEADROOM_GIB" \
        --encode_headroom_gib "$ENCODE_HEADROOM_GIB"
done < "$LIST"

echo "host list done: $n entries $(date -Is)"
