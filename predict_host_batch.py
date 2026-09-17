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
import glob
import os
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
    _, host_k_embeds, host_residues = encode_proteome(
        host_sequences, model, tokenizer, device, args.batch_size, args.max_len,
        desc=f"Encoding host ({host_base})",
    )
    n_host = len(host_sequences)

    for vi, viral_fasta in enumerate(pending, 1):
        out_csv = out_path(viral_fasta)
        name = os.path.splitext(os.path.basename(out_csv))[0]
        print(f"== [{vi}/{len(pending)}] {name}")

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
        raw_predictions = []

        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            for i in range(0, len(inference_tasks), args.batch_size):
                batch = inference_tasks[i : i + args.batch_size]
                pad_q = pad_sequence([viral_residues[q].to(device) for q, _, _ in batch], batch_first=True)
                pad_c = pad_sequence([combined_residues[c].to(device) for _, c, _ in batch], batch_first=True)
                len_q = torch.tensor([len(viral_residues[q]) for q, _, _ in batch], device=device)
                len_c = torch.tensor([len(combined_residues[c]) for _, c, _ in batch], device=device)
                mask_q = (torch.arange(pad_q.shape[1], device=device)[None, :] < len_q[:, None]).long()
                mask_c = (torch.arange(pad_c.shape[1], device=device)[None, :] < len_c[:, None]).long()

                logits, valid_mask = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)
                logits = logits.float().masked_fill(~valid_mask, float("-inf"))
                scores = torch.sigmoid(logits.flatten(1).max(dim=-1).values).cpu().numpy()

                for k, (q_idx, c_idx, is_host) in enumerate(batch):
                    raw_predictions.append((q_idx, c_idx, float(scores[k]), is_host))

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

    print(f"{host_base}: done ({len(pending)} pairs processed).")


if __name__ == "__main__":
    main()
