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


def encode_proteome(sequences, model, tokenizer, device, batch_size, max_len, desc="Encoding",
                    residue_device="cpu", vram_headroom_gib=2.5):
    """Encode a proteome.

    residue_device="cpu" keeps residues off the GPU so that proteome size does
    not drive VRAM. That is the right default on a small card, but when the
    caller is going to park the residues on the GPU anyway it makes every
    tensor cross PCIe twice -- down here, straight back up there, ~3.6 GiB each
    way for a 4933-protein proteome. Pass "cuda" to leave them where they were
    produced.

    "cuda" is a request, not a promise. A 4933-protein proteome is ~3.6 GiB of
    fp16 residues on top of a 2.7 GiB fp32 model and autocast's fp16 weight
    cache, which fits a 48 GiB card and does not fit a 12 GiB one: asking for it
    unconditionally OOMs mid-encode, inside the PLM's swiglu, with the residues
    already banked. So watch free VRAM and spill the remainder to the host once
    it drops under vram_headroom_gib. The switch is one-way -- free memory only
    falls from here, and flipping back and forth would just relocate tensors
    while the allocator is already tight. Callers must handle a mixed list;
    predict_host_batch.py already does, because .to(device) is a no-op on a
    tensor that is on the device and a copy on one that is not.
    """
    query_embeds, key_embeds, residue_list = [], [], []
    keep_on_device = residue_device == "cuda"
    reserve = vram_headroom_gib * 2**30

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

            # Checked once per batch rather than per protein: mem_get_info is a
            # driver call and a batch only banks ~24 MiB, far less than the
            # headroom being defended.
            if keep_on_device and torch.cuda.mem_get_info()[0] < reserve:
                keep_on_device = False

            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for j, seq_len in enumerate(lengths):
                r = res_embed[j, :int(seq_len), :].half()
                # .clone() so the slice stops pinning the whole batch's
                # activation block alive; without it the allocator cannot
                # release res_embed and VRAM grows with the number of batches.
                residue_list.append(r.clone() if keep_on_device else r.cpu())

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
