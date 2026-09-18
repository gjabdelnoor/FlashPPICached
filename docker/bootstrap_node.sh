#!/usr/bin/env bash
# Bring a fresh rented GPU box up to a working sweep worker, from nothing.
#
# Runs ON the new node. Everything it needs is copied from an existing node.
#
# No private key is ever written to rented disk. Two ways to authenticate the
# pull, in order of preference:
#
#   1. The new node generates its own keypair and its PUBLIC half is appended to
#      the source node's authorized_keys. Only the new box holds that private
#      key, and revoking it is one line on the source. Pass SRC_KEY to use it.
#        ssh new 'ssh-keygen -t ed25519 -f ~/.ssh/pull -N "" -C <name>-pull'
#        ssh src  "echo '<pubkey>' >> ~/.ssh/authorized_keys"
#   2. Agent forwarding, when the same key already opens both boxes:
#        ssh -A -p <port> ubuntu@<new-ip> 'bash -s' < bootstrap_node.sh
#
# Option 1 is what the Thunder Compute pair uses, because the two boxes are on
# different accounts and no single key opens both.
#
# Idempotent: every stage checks for its own output first, so a re-run after a
# dropped connection resumes instead of starting over.
set -euo pipefail

# Rented boxes come and go, so the source node is passed in rather than baked
# in -- same reason sync_results.sh takes VAST_HOST from the environment.
#   SRC=1.2.3.4 SRC_PORT=30552 ssh -A ... 'bash -s' < bootstrap_node.sh
SRC="${SRC:?set SRC to the IP of the node to copy from}"
SRC_PORT="${SRC_PORT:-22}"
SRC_USER="${SRC_USER:-ubuntu}"
# SRC_ROOT is probed, not assumed. The fleet has two roots -- the hand-built
# first box keeps everything under projects/ppi, boxes built by this script use
# a flat ~/ppi -- and the wrong guess fails after the venv build with an rsync
# change_dir error that reads like a permissions problem and is not one.
SRC_ROOT="${SRC_ROOT:-}"
DEST="${DEST:-$HOME/ppi}"
# Weights are the one thing that must be bit-identical across boxes: a model
# pulled from HuggingFace instead of from the running node is a different
# experiment, not a faster download. Checked, not assumed.
MODEL_SHA="${MODEL_SHA:-783abc99f0d39c350d9be2e553dbf407b9ebc0fa1d288b31f418b4a3ef223f2c}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.12}"

SRC_KEY="${SRC_KEY:-}"
RSH="ssh -p $SRC_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
[ -n "$SRC_KEY" ] && RSH="$RSH -i $SRC_KEY -o IdentitiesOnly=yes"

# Probe before anything expensive: one round trip that finds the payload root on
# the far side. Doing this up front means a wrong SRC_ROOT costs one second
# instead of a full venv build followed by a confusing rsync failure.
if [ -z "$SRC_ROOT" ]; then
    for cand in /home/ubuntu/ppi /home/ubuntu/projects/ppi "$HOME/ppi"; do
        # -n / BatchMode: without a tty ssh will still try to prompt for a
        # password, and under nohup there is nothing to prompt on, so the probe
        # would fail for a reason that has nothing to do with the payload. The
        # error is captured rather than discarded so a real failure is readable.
        out=$(ssh -n -o BatchMode=yes $RSH "$SRC_USER@$SRC" \
                "test -d $cand/models/flashppi -o -d $cand/flashppi-proteome/models/flashppi" 2>&1)
        if [ $? -eq 0 ]; then
            SRC_ROOT="$cand"
            break
        fi
        [ -n "$out" ] && echo "[bootstrap] probe $cand: $out"
    done
fi
if [ -z "$SRC_ROOT" ]; then
    echo "[bootstrap] FATAL: no payload root found on $SRC_USER@$SRC" >&2
    echo "  tried: /home/ubuntu/ppi /home/ubuntu/projects/ppi $HOME/ppi" >&2
    echo "  pass SRC_ROOT=... explicitly if it lives somewhere else" >&2
    exit 1
fi
REMOTE="$SRC_USER@$SRC"

mkdir -p "$DEST" "$DEST/logs"
cd "$DEST"

say() { echo "[bootstrap] $*"; }

# ---------------------------------------------------------------- environment
if ! command -v uv >/dev/null 2>&1; then
    say "installing uv"
    curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -x "$DEST/venv/bin/python" ]; then
    say "building venv (python $PYTHON_VERSION)"
    uv python install "$PYTHON_VERSION" >/dev/null
    uv venv --python "$PYTHON_VERSION" "$DEST/venv" >/dev/null
fi

# ------------------------------------------------------------------- payload
# Sizes as of this writing: model 2.8G, data 3.4G, viral_cache 5.1G. The viral
# cache is by far the largest and also the most valuable -- recomputing it means
# re-encoding 95 proteomes on a cold box.
#
# Two layouts exist across the fleet, so detect rather than assume. The first
# box was built by hand, with everything under flashppi-proteome/ and the repo
# in a nested directory; boxes built by this script use the flat layout below.
# Guessing wrong fails 30 seconds in, after the venv build, with an rsync
# change_dir error that reads like a permissions problem and is not one.
if ssh $RSH "$REMOTE" "test -d $SRC_ROOT/flashppi-proteome/models/flashppi" 2>/dev/null; then
    SRC_MODEL="$SRC_ROOT/flashppi-proteome/models/flashppi"
    SRC_REPO="$SRC_ROOT/flashppi-proteome/repo"
    say "source layout: nested (flashppi-proteome/)"
else
    SRC_MODEL="$SRC_ROOT/models/flashppi"
    SRC_REPO="$SRC_ROOT/repo"
    say "source layout: flat"
fi

say "pulling payload from $REMOTE:$SRC_PORT"
mkdir -p "$DEST/models" "$DEST/repo"
rsync -az --info=progress2 -e "$RSH" \
    "$REMOTE:$SRC_MODEL" "$DEST/models/"
rsync -az --info=progress2 -e "$RSH" \
    "$REMOTE:$SRC_ROOT/data" "$DEST/"
rsync -az --info=progress2 -e "$RSH" \
    "$REMOTE:$SRC_ROOT/viral_cache" "$DEST/"
rsync -az -e "$RSH" \
    "$REMOTE:$SRC_REPO/" "$DEST/repo/"
# Existing results, so this box skips hosts the others already finished.
mkdir -p "$DEST/output"
rsync -az --ignore-existing --include='*.csv' --exclude='*' -e "$RSH" \
    "$REMOTE:$SRC_ROOT/output/" "$DEST/output/"

got=$(sha256sum "$DEST/models/flashppi/model.safetensors" | cut -d' ' -f1)
if [ "$got" != "$MODEL_SHA" ]; then
    echo "[bootstrap] FATAL: model.safetensors sha mismatch" >&2
    echo "  expected $MODEL_SHA" >&2
    echo "  got      $got" >&2
    exit 1
fi
say "model sha verified"

# --------------------------------------------------------------------- wheels
# uv blocks forever if it is writing progress to a pipe whose reader is gone,
# which is what happens when the launching ssh session dies mid-install. Under
# tmux with output redirected to a file there is no pipe to lose. (Cost 25
# minutes the first time, with uv sitting at 0% CPU in futex_wait.)
if ! "$DEST/venv/bin/python" -c "import torch, faiss, transformers" 2>/dev/null; then
    # tmux is the reason this survives. The base Thunder image ships without it,
    # and plain `nohup ... &` still dies with the ssh session that launched it
    # if the box is a container whose PID 1 tears down the process group.
    command -v tmux >/dev/null 2>&1 || {
        say "installing tmux"
        sudo -n apt-get update -qq >/dev/null 2>&1 || true
        sudo -n apt-get install -y tmux >/dev/null 2>&1 || true
    }
    say "installing wheels (tmux session 'build', log $DEST/logs/build.log)"
    rsync -az -e "$RSH" "$REMOTE:$SRC_ROOT/docker/requirements.lock.txt" "$DEST/" 2>/dev/null \
        || rsync -az -e "$RSH" "$REMOTE:$SRC_ROOT/requirements.lock.txt" "$DEST/"
    tmux kill-session -t build 2>/dev/null || true
    tmux new-session -d -s build \
        "uv pip install --python $DEST/venv/bin/python \
            --index-strategy unsafe-best-match \
            --extra-index-url https://download.pytorch.org/whl/cu130 \
            -r $DEST/requirements.lock.txt > $DEST/logs/build.log 2>&1; \
         echo BUILD_EXIT=\$? >> $DEST/logs/build.log"
    until grep -q BUILD_EXIT "$DEST/logs/build.log" 2>/dev/null; do sleep 5; done
    grep -q 'BUILD_EXIT=0' "$DEST/logs/build.log" || { tail -20 "$DEST/logs/build.log"; exit 1; }
fi

"$DEST/venv/bin/python" - <<'PY'
import torch, transformers, faiss
print(f"[bootstrap] torch {torch.__version__} cuda {torch.version.cuda} "
      f"transformers {transformers.__version__} gpus {torch.cuda.device_count()}")
PY

say "ready. launch with:  START=<idx> bash $DEST/restart_forward.sh"
