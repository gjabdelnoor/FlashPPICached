"""Shared helpers for host/viral proteome embedding caching.

encode_proteome is copied verbatim from predict_cross_proteome.py (same
model calls, same batching) -- no change to how the model is invoked.
The only new behavior is: viral proteome encodings are persisted to disk
so they are computed once instead of once per (host, viral) pair.
"""
import os

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm


def encode_proteome(sequences, model, tokenizer, device, batch_size, max_len, desc="Encoding"):
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
                residue_list.append(res_embed[j, :int(seq_len), :].cpu().half())

    return (
        np.concatenate(query_embeds, axis=0),
        np.concatenate(key_embeds, axis=0),
        residue_list,
    )


def load_fasta(path):
    records = list(SeqIO.parse(path, "fasta"))
    return [str(r.seq) for r in records], [r.id for r in records]


def atomic_save(obj, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_or_compute_viral(viral_fasta, cache_dir, model, tokenizer, device, batch_size, max_len):
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(viral_fasta))[0]
    cache_path = os.path.join(cache_dir, base + ".pt")

    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=False)
        return data["q_embeds"], data["k_embeds"], data["residues"], data["ids"]

    sequences, ids = load_fasta(viral_fasta)
    q_embeds, k_embeds, residues = encode_proteome(
        sequences, model, tokenizer, device, batch_size, max_len, desc=f"Encoding viral ({base})",
    )
    atomic_save({"q_embeds": q_embeds, "k_embeds": k_embeds, "residues": residues, "ids": ids}, cache_path)
    return q_embeds, k_embeds, residues, ids
