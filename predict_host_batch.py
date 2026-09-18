#!/usr/bin/env python3
"""
Process ONE host proteome against every viral proteome in viral_dir,
in a single process.

This replaces calling predict_cross_proteome.py once per (host, viral)
pair. That approach reloaded the model and re-encoded the FULL host
proteome from scratch on every single pair invocation -- ~95x redundant
work per host, since the same host pairs with all 95 viral files.

Here the host is encoded once and kept in memory for the whole run.
Viral embeddings are loaded from an on-disk cache (see
warm_viral_cache.py) instead of being re-encoded per pair. The
FAISS retrieval + contact-prediction logic per pair is unchanged from
predict_cross_proteome.py -- only the redundant encoding is removed.
"""
import argparse
import gc
import glob
import os
import time
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from cache_utils import encode_proteome, load_fasta, load_or_compute_viral


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host_fasta", required=True)
    p.add_argument("--viral_dir", required=True)
    p.add_argument("--viral_cache_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--stage1_top_k", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=1024)
    # Residues are cached on the CPU so that host proteome size does not drive
    # VRAM (see cache_utils.py). That costs a host-to-device copy per tensor per
    # batch, and the same host protein is copied again every time it comes back
    # as a candidate -- with 100 candidates per query that is hundreds of
    # transfers of the same few hundred KB. On a card with room to spare, park
    # the residues on the GPU once instead: ~3.6 GiB for a 4933-protein
    # proteome, after which the .to(device) calls in the batch loop are no-ops.
    p.add_argument("--residue_device", choices=["cpu", "cuda"], default="cpu")
    # Two headrooms, because the two phases defend against different peaks and a
    # single number cannot serve both. Encoding runs one PLM forward at a time
    # and needs little slack, so a small reserve there keeps the maximum number
    # of host residues on the device -- which is where the 2.8x came from. The
    # contact loop runs a 2-layer transformer over padded pairs and peaks around
    # 3 GiB, so the viral preload has to stop well short of full. Setting the
    # encode reserve as high as the contact one would spill nearly every host
    # residue to RAM on a 12 GiB card and give the speedup straight back.
    p.add_argument("--encode_headroom_gib", type=float, default=2.5)
    p.add_argument("--vram_headroom_gib", type=float, default=5.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    host_base = os.path.splitext(os.path.basename(args.host_fasta))[0]
    viral_files = sorted(glob.glob(os.path.join(args.viral_dir, "*.faa")))

    def out_path(viral_fasta):
        viral_base = os.path.splitext(os.path.basename(viral_fasta))[0]
        return os.path.join(args.output_dir, f"{host_base}__{viral_base}.csv")

    pending = [vf for vf in viral_files if not os.path.exists(out_path(vf))]
    if not pending:
        print(f"{host_base}: all {len(viral_files)} viral pairs already done, skipping.")
        return

    print(f"Loading model ({args.model_name}) to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    print(f"Encoding host {host_base}...")
    host_sequences, host_ids = load_fasta(args.host_fasta)
    res_dev = args.residue_device if torch.cuda.is_available() else "cpu"
    _, host_k_embeds, host_residues = encode_proteome(
        host_sequences, model, tokenizer, device, args.batch_size, args.max_len,
        desc=f"Encoding host ({host_base})", residue_device=res_dev,
        vram_headroom_gib=args.encode_headroom_gib,
    )
    n_host = len(host_sequences)

    if res_dev == "cuda":
        on_dev = [r for r in host_residues if r.is_cuda]
        gib = sum(r.numel() * r.element_size() for r in on_dev) / 2**30
        print(
            f"Host residues resident on {device}: {gib:.2f} GiB across "
            f"{len(on_dev)}/{len(host_residues)} proteins "
            f"(rest spilled to host RAM, headroom {args.vram_headroom_gib:.1f} GiB)",
            flush=True,
        )

    # With residues on the device, the remaining per-pair transfer is the viral
    # cache itself: a torch.load off disk plus a host-to-device copy, repeated
    # for all 95 viral proteomes on every host. They are small -- ~35 MiB each,
    # ~3.2 GiB for the full set -- so stage them once and the inner loop touches
    # disk and PCIe zero times.
    #
    # How many fit is a property of the card, not of the run. A 48 GiB A6000
    # holds every one of them; a 12 GiB card holds the host residues and little
    # else, and preloading all 95 there OOMs partway through the first pair.
    # So measure the free VRAM that is actually left after the host is encoded,
    # keep back enough for activations, and fill the rest. Whatever does not fit
    # falls through to the per-pair load below -- correct either way, just with a
    # disk read on the pairs that missed.
    viral_cache = {}
    if res_dev == "cuda":
        budget = torch.cuda.mem_get_info()[0] - args.vram_headroom_gib * 2**30
        used = 0
        for viral_fasta in pending:
            vq, vk, vres, vids = load_or_compute_viral(
                viral_fasta, args.viral_cache_dir, model, tokenizer, device,
                args.batch_size, args.max_len,
            )
            need = sum(r.numel() * r.element_size() for r in vres)
            if used + need > budget:
                break
            viral_cache[viral_fasta] = (vq, vk, [r.to(device) for r in vres], vids)
            used += need
        print(
            f"Viral residues resident on {device}: {used/2**30:.2f} GiB across "
            f"{len(viral_cache)}/{len(pending)} proteomes "
            f"(budget {max(budget, 0)/2**30:.2f} GiB)",
            flush=True,
        )

    for vi, viral_fasta in enumerate(pending, 1):
        out_csv = out_path(viral_fasta)
        name = os.path.splitext(os.path.basename(out_csv))[0]
        print(f"== [{vi}/{len(pending)}] {name}", flush=True)
        t_pair = time.time()

        if viral_fasta in viral_cache:
            viral_q_embeds, viral_k_embeds, viral_residues, viral_ids = viral_cache[viral_fasta]
        else:
            viral_q_embeds, viral_k_embeds, viral_residues, viral_ids = load_or_compute_viral(
                viral_fasta, args.viral_cache_dir, model, tokenizer, device, args.batch_size, args.max_len,
            )
        n_viral = len(viral_ids)

        combined_k_embeds = np.concatenate([host_k_embeds, viral_k_embeds], axis=0)
        index = faiss.IndexFlatIP(combined_k_embeds.shape[1])
        index.add(combined_k_embeds)

        search_k = min(args.stage1_top_k + 1, len(combined_k_embeds))
        _, I = index.search(viral_q_embeds, k=search_k)

        inference_tasks = []
        for q_idx in range(n_viral):
            for c_idx in I[q_idx]:
                if c_idx < 0 or c_idx == n_host + q_idx:
                    continue
                inference_tasks.append((q_idx, int(c_idx), c_idx < n_host))

        combined_residues = host_residues + viral_residues
        task_order = {(q, c): i for i, (q, c, _) in enumerate(inference_tasks)}
        inference_tasks.sort(key=lambda t: (len(viral_residues[t[0]]), len(combined_residues[t[1]])))

        # Everything the batch loop touches is staged on the device up front, so
        # nothing crosses PCIe once the loop starts. Three separate crossings
        # used to happen per batch, and residues were only the largest:
        #   1. one .to(device) per residue tensor, 2 x batch_size of them, with
        #      the same host protein re-sent every time it came back as a
        #      candidate -- hundreds of times per pair at top_k=100;
        #   2. len_q / len_c built from Python lists, a small H2D copy that also
        #      forced a sync;
        #   3. .cpu() on every batch's scores, a D2H that stalled the queue
        #      before the next batch could be enqueued.
        # Now the lengths live on the device as one tensor gathered by device-
        # side index tensors, and scores accumulate on the device for a single
        # transfer per pair.
        len_viral_all = torch.tensor([len(r) for r in viral_residues], device=device)
        len_comb_all = torch.tensor([len(r) for r in combined_residues], device=device)
        q_idx_all = torch.tensor([t[0] for t in inference_tasks], device=device)
        c_idx_all = torch.tensor([t[1] for t in inference_tasks], device=device)
        score_chunks = []
        n_tasks = len(inference_tasks)

        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            for i in range(0, len(inference_tasks), args.batch_size):
                batch = inference_tasks[i : i + args.batch_size]
                # .to(device) is a no-op returning self when the tensor is
                # already resident, so this stays correct under
                # --residue_device cpu without costing a copy under cuda.
                pad_q = pad_sequence([viral_residues[q].to(device) for q, _, _ in batch], batch_first=True)
                pad_c = pad_sequence([combined_residues[c].to(device) for _, c, _ in batch], batch_first=True)
                len_q = len_viral_all[q_idx_all[i : i + len(batch)]]
                len_c = len_comb_all[c_idx_all[i : i + len(batch)]]
                mask_q = (torch.arange(pad_q.shape[1], device=device)[None, :] < len_q[:, None]).long()
                mask_c = (torch.arange(pad_c.shape[1], device=device)[None, :] < len_c[:, None]).long()

                logits, valid_mask = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)
                logits = logits.float().masked_fill(~valid_mask, float("-inf"))
                score_chunks.append(torch.sigmoid(logits.flatten(1).max(dim=-1).values))

        scores = torch.cat(score_chunks).cpu().numpy()
        raw_predictions = [
            (q_idx, c_idx, float(scores[k]), is_host)
            for k, (q_idx, c_idx, is_host) in enumerate(inference_tasks)
        ]

        raw_predictions.sort(key=lambda p: task_order[p[:2]])
        query_predictions = defaultdict(list)
        for q_idx, c_idx, score, is_host in raw_predictions:
            query_predictions[q_idx].append((c_idx, score, is_host))

        results = []
        for q_idx, preds in query_predictions.items():
            host_preds = [(c, s) for c, s, h in preds if h]
            if not host_preds:
                continue
            best_host_c_idx, best_host_score = max(host_preds, key=lambda x: x[1])
            if best_host_score <= args.threshold:
                continue
            host_is_best_contact = max(preds, key=lambda x: x[1])[2]
            results.append({
                "viral_id": viral_ids[q_idx],
                "host_id": host_ids[best_host_c_idx],
                "contact_score": best_host_score,
                "host_is_best_contact": host_is_best_contact,
            })

        df = pd.DataFrame(results)
        if df.empty:
            df = pd.DataFrame(columns=["viral_id", "host_id", "contact_score", "host_is_best_contact"])
        else:
            df = df.sort_values("contact_score", ascending=False)
        df.to_csv(out_csv, index=False)

        # Reclaim between pairs. Length bucketing hands the allocator a new
        # shape almost every batch, so blocks cached during one pair rarely fit
        # the next one's requests; with 48 GB free nothing forces a reclaim and
        # the reserved pool drifts upward across all 95 pairs even though the
        # live set is flat. Dropping the per-pair tensors explicitly and then
        # reclaiming keeps reserved pinned to the resident working set.
        del score_chunks, scores, len_viral_all, len_comb_all, q_idx_all, c_idx_all
        del raw_predictions, query_predictions, combined_residues
        gc.collect()
        torch.cuda.empty_cache()

        # Per-pair wall clock plus the allocator high-water marks, so throughput
        # and drift are both readable straight from the log. flush because print
        # to a redirected file is block-buffered and otherwise shows nothing.
        print(
            f"   {n_tasks} tasks in {time.time() - t_pair:.1f}s | "
            f"alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB "
            f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB",
            flush=True,
        )

    print(f"{host_base}: done ({len(pending)} pairs processed).")


if __name__ == "__main__":
    main()
