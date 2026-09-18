#!/usr/bin/env bash
# Merge sweep results from every compute box into the capsid MinIO bucket.
#
# No compute box can reach capsid -- UPRM blocks :22 and :9000 inbound, and the
# rented boxes cannot even resolve its name -- so this laptop is the hop. That
# is why the sync runs here and not on a worker: a box that cannot see capsid
# cannot push to it, however the loop is written.
#
# The merge is bidirectional on purpose. Boxes walk the host list from
# different starting points and converge somewhere in the middle. Pushing the
# union back to every box means each one's "has this pair got a CSV already?"
# check also sees the other boxes' work, so they never redo the same host.
#
# Safe to run repeatedly while the sweeps are live: rsync moves only new CSVs
# and mc mirror uploads only what changed.
set -euo pipefail

STAGING="${STAGING:-$HOME/projects/ppi/results-staging}"
BUCKET="${BUCKET:-local/flashppi-results}"
TOTAL_PAIRS="${TOTAL_PAIRS:-38285}"

# Node list, "<ssh-host>:<remote-output-dir>". Rented boxes come and go, so this
# is data, not code -- override with NODES="..." rather than editing the loop.
# `thunder` and `thunder2` are ssh-config aliases, so their IPs and ports live
# in ~/.ssh/config and nothing here goes stale when an instance is replaced.
NODES="${NODES:-thunder:/home/ubuntu/projects/ppi/output thunder2:/home/ubuntu/ppi/output thunder3:/home/ubuntu/ppi/output}"

mkdir -p "$STAGING"

# Only completed CSVs. The .log files are still being appended by live workers,
# so including them would force a resync on every pass.
i=0
for node in $NODES; do
    host="${node%%:*}"
    dir="${node#*:}"
    i=$((i + 1))
    if ssh -o ConnectTimeout=15 -o BatchMode=yes "$host" true 2>/dev/null; then
        echo "[$i] $host -> staging"
        rsync -az --include='*.csv' --exclude='*' "$host:$dir/" "$STAGING/"
    else
        echo "[$i] $host unreachable, skipping"
    fi
done

# --ignore-existing: never overwrite a box's own fresh output with an older copy.
for node in $NODES; do
    host="${node%%:*}"
    dir="${node#*:}"
    ssh -o ConnectTimeout=15 -o BatchMode=yes "$host" true 2>/dev/null || continue
    rsync -az --ignore-existing "$STAGING/" "$host:$dir/" 2>/dev/null \
        || echo "  ($host push failed, continuing)"
done

echo "[+] staging -> capsid -> MinIO ($BUCKET)"
ssh capsid 'mkdir -p ~/ppi-results'
rsync -az "$STAGING/" capsid:~/ppi-results/
ssh capsid "export PATH=\$HOME/.local/bin:\$PATH
            mc mb --ignore-existing $BUCKET >/dev/null
            mc mirror --overwrite --quiet ~/ppi-results/ $BUCKET/ >/dev/null"

c=$(find "$STAGING" -name '*.csv' | wc -l)
echo "synced $c result CSVs -> $BUCKET  ($(( c * 100 / TOTAL_PAIRS ))% of $TOTAL_PAIRS pairs)"
