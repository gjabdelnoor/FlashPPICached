#!/usr/bin/env bash
# Run sync_results.sh on a timer, so results reach capsid without anyone
# remembering to run it. Lives on the laptop because the laptop is the only hop
# that can reach both the compute boxes and capsid.
#
# Each pass is short (incremental rsync of a few hundred small CSVs), so the
# loop just sleeps between passes rather than trying to overlap them: two
# concurrent mc mirror runs against the same bucket would race, and there is
# nothing to gain from it.
#
# Usage:  sync_daemon.sh [interval_seconds]      (default 600)
#         stop with:  touch ~/projects/ppi/.sync-stop
set -uo pipefail

INTERVAL="${1:-600}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOPFILE="${STOPFILE:-$HOME/projects/ppi/.sync-stop}"
LOG="${LOG:-$HOME/projects/ppi/sync.log}"

mkdir -p "$(dirname "$LOG")"
rm -f "$STOPFILE"

echo "$(date -Is) sync daemon up, interval ${INTERVAL}s" >> "$LOG"
while [ ! -f "$STOPFILE" ]; do
    # A failed pass must not kill the loop: the boxes bounce (a rented instance
    # is replaced, ssh times out) and the next pass picks up where it left off.
    # sync_results.sh already tolerates an unreachable node; this catches the
    # rest, including capsid being briefly unavailable.
    if bash "$HERE/sync_results.sh" >> "$LOG" 2>&1; then
        echo "$(date -Is) pass ok" >> "$LOG"
    else
        echo "$(date -Is) pass FAILED (continuing)" >> "$LOG"
    fi
    sleep "$INTERVAL"
done
echo "$(date -Is) sync daemon stopped" >> "$LOG"
