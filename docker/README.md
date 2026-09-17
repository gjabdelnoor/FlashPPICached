# Proteome / viral-host PPI images

Two images, layered. `proteome-base` is reusable; `flashppi-vhppi` is the frozen
state of the September 2026 viral-host sweep.

| Image | Contents | Size (docker) | Size (podman) |
|---|---|---|---|
| `proteome-base:cu130` | CUDA 13.0 runtime, Python 3.12.12, torch 2.14.0+cu130, transformers 5.17.0, faiss-cpu 1.15.1, biopython 1.88, pandas 3.0.5 | 4.90 GB | 8.81 GB |
| `flashppi-vhppi:current` | the above, plus the pipeline code and the 2.8 GB FlashPPI checkpoint | 7.62 GB | 11.74 GB |

The two engines disagree on size because they lay the image out differently
(19 layers under docker, 17 under podman), not because the contents differ —
both builds resolve to the same 84 distributions, sha256 `8bb1651d…` over the
sorted `name==version` list. Docker-to-docker the image is byte-identical:
`proteome-base` and `flashppi-vhppi` inspect to the same size and layer digests
on the build laptop and after `docker load` on another host.

Every version is pinned in `requirements.lock.txt`, captured from the RTX A6000
host on 2026-09-17. `uv` pins the interpreter too, so a rebuild reproduces the
same Python rather than whatever Ubuntu currently ships.

One deviation from the source box: it ran Python 3.12.13, which
python-build-standalone does not publish. The image pins 3.12.12, its newest
3.12. Same cp312 ABI, so every wheel in the lock file is bit-identical either
way. Override with `--build-arg PYTHON_VERSION=...` if that ever matters.

## Where the artifacts live

Both images are stored on the capsid MinIO as one layer-deduped archive
(11 GiB, sha256 `333b671a…`, verified by streaming it back out):

```sh
mc cp local/flashppi-images/proteome-images.tar .
sha256sum -c <(mc cat local/flashppi-images/proteome-images.tar.sha256)
docker load -i proteome-images.tar     # restores BOTH tags
```

`local/flashppi-images/build-context/` holds the Dockerfiles, lock file and
pipeline source, so the images can be rebuilt from scratch without this repo.
Sweep results land in `local/flashppi-results/`; `sync_results.sh` refreshes
them.

Build note: the images were built twice — on capsid with podman and on the
laptop with docker — never on thunder. Thunder's Docker is a Thunder Compute
shim (`Server Version: library-import`, no BuildKit) on a host where `unshare`
is blocked, so it can *run* images but cannot build them.

## Verified

- **Package set**: 84 distributions, identical between the docker and podman
  builds (sha256 `8bb1651d…`). Note `/opt/venv/bin/pip` does not exist — `uv
  venv` does not install pip — so use `uv pip freeze --python
  /opt/venv/bin/python` or `importlib.metadata` to inspect the environment.
- **GPU path**: verified on an RTX 3060 (driver 595.84, nvidia-container-toolkit
  1.20.0). `version` reports `cuda avail True`, and a real `pair` run —
  300 BL21 proteins × 58 T7LD proteins, 322 candidates — produced 48 scored
  interactions (top `T7LD_proteome_20 ↔ BL21_proteome_258` at 0.333) in under
  30 s. Not merely device detection: FAISS retrieval and contact prediction both
  ran on the card, and it released to 32 MiB on exit.
- **Round-trip**: the MinIO archive was streamed back out and its sha256
  rechecked (`333b671a…`).

The 3060 needs a smaller working set than the A6000 the sweep runs on: use
`NUM_SHARDS=1` and a reduced `BATCH_SIZE`, and expect a full 4,000-protein host
proteome not to fit in 12 GB.

## Build

```sh
docker build -f Dockerfile.base  -t proteome-base:cu130   .
docker build -f Dockerfile.vhppi -t flashppi-vhppi:current .
```

The build context must contain `models/flashppi/` (the checkpoint). On the
thunder box that directory is hardlinked in, which costs no extra disk:

```sh
cp -al ../flashppi-proteome/models/flashppi models/flashppi
```

## Run the sweep

```sh
docker run --gpus all \
  -v /path/to/data:/data:ro \
  -v /path/to/viral_cache:/cache \
  -v /path/to/output:/output \
  flashppi-vhppi:current
```

`/data` must hold `proteomes/*.faa` (hosts) and `viral_proteomes/*.faa`.
If `/cache` is empty the viral embedding cache is built first (~6 min, 5.1 GB
for 95 viral proteomes); otherwise it is reused.

Results land in `/output` as one `<host>__<viral>.csv` per pair, plus a log per
shard. A pair whose CSV already exists is skipped, so the sweep is resumable —
kill it and rerun the identical command.

### Other subcommands

```sh
docker run --gpus all flashppi-vhppi:current version        # print the pinned stack
docker run --gpus all ... flashppi-vhppi:current warm-cache # build the viral cache only
docker run --gpus all ... flashppi-vhppi:current pair \
    --host_fasta /data/proteomes/X.faa \
    --viral_fasta /data/viral_proteomes/Y.faa \
    --model_name /opt/models/flashppi --output /output/xy.csv
docker run --gpus all ... flashppi-vhppi:current bash       # anything else runs as-is
```

## Tuning

| Variable | Default | Notes |
|---|---|---|
| `NUM_SHARDS` | `3` | Parallel host workers in one container. Budget ~9 GiB of VRAM and ~5 GiB of RAM each; three fit a 48 GB A6000 with room to spare. Raising it past the core count is counterproductive — the per-pair CPU work is what fills the GPU's idle gaps. |
| `BATCH_SIZE` | `32` | Benchmarked optimum. Measured on the A6000 over all 30,792 candidate pairs of one host×viral: 8 → 41.0 s, 32 → 31.5 s, 64 → 37.4 s, 128 → 48.1 s (allocator pressure). |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Set by the entrypoint; override by passing it in. See below — without it a shard drifts to ~3× its working set. |

### Why VRAM is not what it looks like

A shard's real cost is small and fixed: model weights are **2.78 GiB**, and
between batches nothing else is live. Residue embeddings never reach the GPU —
`cache_utils.encode_proteome` ends every tensor with `.cpu().half()` — so host
proteome size does not affect VRAM at all. The transient peak comes from
`predict_contacts` and scales with `BATCH_SIZE` (~0.10 GiB per unit at
`max_len=1024`), reaching ~8 GiB reserved at 32. Encoding peaks lower, ~6 GiB.

What the card *reports* is much larger, because length bucketing gives each
batch a different shape and the caching allocator ends up holding blocks that
never fit the next request. Replaying 300 realistic bucketed batches:

| | allocated | reserved | ratio |
|---|---|---|---|
| default | 2.79 GiB | 10.53 GiB | 3.77× |
| `expandable_segments:True` | 2.74 GiB | 5.32 GiB | 1.94× |

Peak reserved drops from 11.39 GiB to 6.55 GiB, and the repeated forced
reclaims vanish. Without it, a shard on a large card simply grows until it owns
its share — which is why two shards appeared to need 47 GB when they needed
about 16.

`BATCH_SIZE` is only optimal because the candidate pairs are length-bucketed
before batching (`predict_host_batch.py`). Without that, padding waste grows
with the batch and 8 wins instead.

## Throughput

On one A6000 with `NUM_SHARDS=2`: ~24.7 s per host×viral pair combined, plus
~3.7 min to encode each host proteome once. The full 403 × 95 = 38,285-pair
sweep takes roughly 11–12 days.

## Reproducing the exact September 2026 run

The pipeline code is baked at `/opt/flashppi`. `predict_cross_proteome.py`
matches `gjabdelnoor/FlashPPICached` commit `d94c6e7` on branch
`cache-host-encoding`; `predict_host_batch.py`, `warm_viral_cache.py` and
`cache_utils.py` carry the same length-bucketing change but live only here.

Numerics note: length bucketing changes which sequences share a batch, so fp16
contact scores shift by up to 3.8e-4 versus the unbucketed order. Across the
full 30,792-pair validation that moved no reported interaction and crossed no
0.4 threshold. The mask rewrite alone is bit-identical.
