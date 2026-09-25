#!/usr/bin/env python3
"""
FlashPPI: Cross-Proteome PPI Screening

Predicts protein-protein interactions between two proteomes (host and viral).

Viral proteins are queries; host proteins are keys.
For each viral protein, candidates are retrieved from a combined key space of
host + viral proteins (excluding self-hits). This lets us determine whether the
best match for a viral protein is a host protein (cross-organism interaction) or
another viral protein (within-organism match), reported as `host_is_best_contact`.

Usage:
  python predict_cross_proteome.py \\
    --host_fasta host.fasta \\
    --viral_fasta virus.fasta \\
    --output predictions.csv
"""
import argparse
import os
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def encode_proteome(sequences, model, tokenizer, device, batch_size, max_len, desc="Encoding"):
    """
    Encode a list of protein sequences.

    Returns:
        query_embeds: np.ndarray (N, D) - query projection embeddings
        key_embeds:   np.ndarray (N, D) - key projection embeddings
        residue_list: list of device (GPU) tensors, each (seq_len, D)
    """
    query_embeds, key_embeds, residue_list = [], [], []

    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        for i in tqdm(range(0, len(sequences), batch_size), desc=desc, unit="batch"):
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            ).to(device)

            res_embed = model.encode_protein(inputs["input_ids"], inputs["attention_mask"])
            query_embeds.append(model.head_q(res_embed, inputs["attention_mask"]).cpu().float().numpy())
            key_embeds.append(model.head_k(res_embed, inputs["attention_mask"]).cpu().float().numpy())

            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for j, seq_len in enumerate(lengths):
                residue_list.append(res_embed[j, :int(seq_len), :].clone())

    return (
        np.concatenate(query_embeds, axis=0),
        np.concatenate(key_embeds, axis=0),
        residue_list,
    )


def main():
    parser = argparse.ArgumentParser(
        description="FlashPPI: Cross-Proteome PPI Screening between two organisms"
    )
    parser.add_argument("--host_fasta", type=str, required=True,
                        help="Path to the host proteome FASTA file (keys).")
    parser.add_argument("--viral_fasta", type=str, required=True,
                        help="Path to the viral/pathogen proteome FASTA file (queries).")
    parser.add_argument("--output", type=str, default="cross_predictions.csv",
                        help="Path to save the output CSV. Default: cross_predictions.csv")
    parser.add_argument("--model_name", type=str, default="tattabio/flashppi",
                        help="HuggingFace model ID or local path.")
    parser.add_argument("--stage1_top_k", type=int, default=100,
                        help="Number of nearest neighbors to retrieve per viral protein in stage 1.")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Contact score threshold to keep predictions.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for model inference.")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="Maximum sequence length.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="If set, cache/reuse host and viral proteome encodings here (keyed by filename).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model and tokenizer ({args.model_name}) to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    # Load sequences
    print(f"\nLoading host sequences from {args.host_fasta}...")
    host_records = list(SeqIO.parse(args.host_fasta, "fasta"))
    host_sequences = [str(r.seq) for r in host_records]
    host_ids = [r.id for r in host_records]
    print(f"  {len(host_sequences)} host proteins")

    print(f"Loading viral sequences from {args.viral_fasta}...")
    viral_records = list(SeqIO.parse(args.viral_fasta, "fasta"))
    viral_sequences = [str(r.seq) for r in viral_records]
    viral_ids = [r.id for r in viral_records]
    print(f"  {len(viral_sequences)} viral proteins")

    if not host_sequences:
        raise ValueError("Host FASTA file contains no sequences.")
    if not viral_sequences:
        raise ValueError("Viral FASTA file contains no sequences.")

    # Stage 1: Encode both proteomes
    print("\nStage 1: Encoding proteomes...")
    host_cache = os.path.join(args.cache_dir, os.path.basename(args.host_fasta) + ".pt") if args.cache_dir else None
    if host_cache and os.path.exists(host_cache):
        host_k_embeds, host_residues = torch.load(host_cache, weights_only=False)
    else:
        _, host_k_embeds, host_residues = encode_proteome(
            host_sequences, model, tokenizer, device, args.batch_size, args.max_len,
            desc="Encoding host",
        )
        if host_cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            torch.save((host_k_embeds, host_residues), host_cache)
    viral_cache = os.path.join(args.cache_dir, os.path.basename(args.viral_fasta) + ".pt") if args.cache_dir else None
    if viral_cache and os.path.exists(viral_cache):
        viral_q_embeds, viral_k_embeds, viral_residues = torch.load(viral_cache, weights_only=False)
    else:
        viral_q_embeds, viral_k_embeds, viral_residues = encode_proteome(
            viral_sequences, model, tokenizer, device, args.batch_size, args.max_len,
            desc="Encoding viral",
        )
        if viral_cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            torch.save((viral_q_embeds, viral_k_embeds, viral_residues), viral_cache)

    n_host = len(host_sequences)
    n_viral = len(viral_sequences)

    # Build combined key space with host and viral keys.
    # Searching viral queries against this unified space lets us determine whether a viral
    # protein's best match is a host protein (cross-organism) or another viral protein
    # (within-organism), reported as `host_is_best_contact`.
    combined_k_embeds = np.concatenate([host_k_embeds, viral_k_embeds], axis=0)

    print(f"\nFAISS Retrieval (combined key space: {n_host} host + {n_viral} viral = {len(combined_k_embeds)})...")
    index = faiss.IndexFlatIP(combined_k_embeds.shape[1])
    index.add(combined_k_embeds)

    search_k = min(args.stage1_top_k + 1, len(combined_k_embeds))
    _, I = index.search(viral_q_embeds, k=search_k)

    # Build inference tasks, skipping self-hits (viral protein q_idx → combined index n_host + q_idx)
    inference_tasks = []  # (viral_idx, combined_idx, is_host)
    for q_idx in range(n_viral):
        for c_idx in I[q_idx]:
            if c_idx < 0 or c_idx == n_host + q_idx:
                continue
            inference_tasks.append((q_idx, int(c_idx), c_idx < n_host))

    print(f"Found {len(inference_tasks)} candidate pairs. Starting contact prediction...")

    # Stage 2: Fine-grained contact prediction
    model.plm.cpu(); model.head_q.cpu(); model.head_k.cpu()
    combined_residues = host_residues + viral_residues
    task_order = {(q, c): i for i, (q, c, _) in enumerate(inference_tasks)}
    inference_tasks.sort(key=lambda t: (len(viral_residues[t[0]]), len(combined_residues[t[1]])))

    raw_predictions = []  # (viral_idx, combined_idx, contact_score, is_host)

    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        for i in tqdm(range(0, len(inference_tasks), args.batch_size),
                      desc="Contact prediction", unit="batch"):
            batch = inference_tasks[i : i + args.batch_size]

            pad_q = pad_sequence([viral_residues[q].to(device) for q, _, _ in batch], batch_first=True)
            pad_c = pad_sequence([combined_residues[c].to(device) for _, c, _ in batch], batch_first=True)
            len_q = torch.tensor([len(viral_residues[q]) for q, _, _ in batch], device=device)
            len_c = torch.tensor([len(combined_residues[c]) for _, c, _ in batch], device=device)
            mask_q = (torch.arange(pad_q.shape[1], device=device)[None, :] < len_q[:, None]).long()
            mask_c = (torch.arange(pad_c.shape[1], device=device)[None, :] < len_c[:, None]).long()

            logits, valid_mask = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)
            # exclude padded cells from the max, as FlashPPIModel.forward does
            logits = logits.float().masked_fill(~valid_mask, float("-inf"))
            scores = torch.sigmoid(logits.flatten(1).max(dim=-1).values).cpu().numpy()

            for k, (q_idx, c_idx, is_host) in enumerate(batch):
                raw_predictions.append((q_idx, c_idx, float(scores[k]), is_host))

    # Group by viral protein: pick best host match, annotate host_is_best_contact
    raw_predictions.sort(key=lambda p: task_order[p[:2]])  # Preserve FAISS tie-breaking and CSV order.
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
    if not df.empty:
        df = df.sort_values("contact_score", ascending=False)
        df.to_csv(args.output, index=False)
        print(f"\nSuccess! Saved {len(df)} predictions to {args.output}")
    else:
        print(f"\nNo cross-proteome interactions found above the threshold of {args.threshold}.")


if __name__ == "__main__":
    main()
