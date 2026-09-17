#!/usr/bin/env bash
# Merge sweep results from every compute box into the capsid MinIO bucket.
#
# Neither thunder nor the vast box can reach capsid -- UPRM blocks :22 and :9000
# inbound -- so this laptop is the hop for both.
#
# The merge is bidirectional on purpose. thunder walks the host list upward and
# vast walks it downward; they converge somewhere in the middle. Pushing the
# union back to both means each one's "has this CSV already?" check also sees
# the other's work, so the two never redo the same host when they meet.
#
# Safe to run repeatedly while the sweeps are live: rsync moves only new CSVs
# and mc mirror uploads only what changed.
set -euo pipefail

STAGING="${STAGING:-$HOME/projects/ppi/results-staging}"
BUCKET="${BUCKET:-local/flashppi-results}"
THUNDER_OUT="${THUNDER_OUT:-/home/ubuntu/projects/ppi/output}"
VAST_OUT="${VAST_OUT:-/root/ppi/output}"
# Rented boxes come and go; set these per run rather than editing the script.
#   VAST_HOST=root@1.2.3.4 VAST_RSH='ssh -p 12345' ./sync_results.sh
VAST_RSH="${VAST_RSH:-ssh}"
VAST_HOST="${VAST_HOST:-}"

mkdir -p "$STAGING"

# Only completed CSVs. The .log files are still being appended by live shards,
# so including them would force a resync on every pass.
echo "[1/5] thunder -> staging"
rsync -az --include='*.csv' --exclude='*' "thunder:$THUNDER_OUT/" "$STAGING/"

echo "[2/5] vast -> staging"
rsync -az -e "$VAST_RSH" --include='*.csv' --exclude='*' \
    "$VAST_HOST:$VAST_OUT/" "$STAGING/" || echo "  (vast unreachable, skipping)"

# --ignore-existing: never overwrite a box's own fresh output with an older copy.
echo "[3/5] staging -> thunder (so it skips vast's hosts)"
rsync -az --ignore-existing "$STAGING/" "thunder:$THUNDER_OUT/"

echo "[4/5] staging -> vast (so it skips thunder's hosts)"
rsync -az --ignore-existing -e "$VAST_RSH" "$STAGING/" "$VAST_HOST:$VAST_OUT/" \
    || echo "  (vast unreachable, skipping)"

echo "[5/5] staging -> capsid -> MinIO ($BUCKET)"
ssh capsid 'mkdir -p ~/ppi-results'
rsync -az "$STAGING/" capsid:~/ppi-results/
ssh capsid "export PATH=\$HOME/.local/bin:\$PATH
            mc mb --ignore-existing $BUCKET >/dev/null
            mc mirror --overwrite --quiet ~/ppi-results/ $BUCKET/ >/dev/null"

n=$(find "$STAGING" -name '*.csv' | wc -l)
echo "synced $n result CSVs -> $BUCKET  ($(( n * 100 / 38285 ))% of 38,285 pairs)"
