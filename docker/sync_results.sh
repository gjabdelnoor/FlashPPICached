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
#
# The sentinel host `self` means "this machine, no ssh" -- needed because the
# box running the sync can also be a compute box, and if its directory is not
# in this list its output never leaves the machine. That happened: the 3060
# produced 190 pairs that sat in ~/projects/ppi/output while the sync reported
# success every ten minutes, because the list only named the three A6000s.
NODES="${NODES:-self:$HOME/projects/ppi/output thunder:/home/ubuntu/projects/ppi/output thunder2:/home/ubuntu/ppi/output thunder3:/home/ubuntu/ppi/output}"

mkdir -p "$STAGING"

# Pull from one node into staging. `self` is read locally; everything else over
# ssh with a reachability pre-check.
pull_node() {
    host="${1%%:*}"; dir="${1#*:}"
    if [ "$host" = "self" ]; then
        [ -d "$dir" ] || { echo "  (self $dir missing, skipping)"; return 0; }
        rsync -az --include='*.csv' --exclude='*' "$dir/" "$STAGING/"
    elif ssh -o ConnectTimeout=15 -o BatchMode=yes "$host" true 2>/dev/null; then
        rsync -az --include='*.csv' --exclude='*' "$host:$dir/" "$STAGING/"
    else
        echo "  ($host unreachable, skipping)"
    fi
}

# Push the union back, so each box's "already done?" check sees every box's
# work. --ignore-existing: never overwrite a box's fresh output with an older copy.
push_node() {
    host="${1%%:*}"; dir="${1#*:}"
    if [ "$host" = "self" ]; then
        [ -d "$dir" ] && rsync -az --ignore-existing "$STAGING/" "$dir/" || true
    elif ssh -o ConnectTimeout=15 -o BatchMode=yes "$host" true 2>/dev/null; then
        rsync -az --ignore-existing "$STAGING/" "$host:$dir/" 2>/dev/null || true
    fi
}

# Only completed CSVs. The .log files are still being appended by live workers,
# so including them would force a resync on every pass.
i=0
for node in $NODES; do
    i=$((i + 1))
    echo "[$i] ${node%%:*} -> staging"
    pull_node "$node"
done

for node in $NODES; do
    push_node "$node"
done

echo "[+] staging -> capsid -> MinIO ($BUCKET)"
ssh capsid 'mkdir -p ~/ppi-results'
rsync -az "$STAGING/" capsid:~/ppi-results/
ssh capsid "export PATH=\$HOME/.local/bin:\$PATH
            mc mb --ignore-existing $BUCKET >/dev/null
            mc mirror --overwrite --quiet ~/ppi-results/ $BUCKET/ >/dev/null"

c=$(find "$STAGING" -name '*.csv' | wc -l)
echo "synced $c result CSVs -> $BUCKET  ($(( c * 100 / TOTAL_PAIRS ))% of $TOTAL_PAIRS pairs)"
