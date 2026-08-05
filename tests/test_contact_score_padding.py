#!/usr/bin/env python
"""contact_score must not depend on batch composition.

Scores one short pair alone, then batched beside a much longer pair, and asserts the two
agree. Fails if valid_mask is not applied before the max. Requires 1 GPU.
"""
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

MODEL = "tattabio/flashppi2"
TOL = 1e-4

SHORT_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
SHORT_B = "MSKQPSDVSSECDREGRQLQPAERPPQLRPGAPTSLQTEPQGNPEGNHGGEGDSCPHGSPQGPLAPPASPGPFATRSP"
LONG = (
    "MAEQVALSRTQVCGILREELFQGDAFHQSDTHIFIIMGASGDLAKKKIYPTIWWLFRDGLLPENTFIVGYARSRLTV"
    "ADIRKQSEPFFKATPEEKLKLEDFFARNSYVAGQYDDAASYQRLNSHMNALHLGSQANRLFYLALPPTVYEAVTKNI"
    "HESCMSQIGWNRIIVEKPFGRDLQSSDRLSNHISSLFREDQIYRIDHYLGKEMVQNLMVLRFANRIFGPIWNRDNIA"
    "CVILTFKEPFGTEGRGGYFDEFGIIRDVMQNHLLQMLCLVAMEKPASTNSDDVRDEKVKVLKCISEVQANNVVLGQY"
    "VGNPDGEGEATKGYLDDPTVPRGSTTATFAAVVLYVENERWDGVPFILRCGKALNERKAEVRLQFHDVAGDIFHQQC"
    "KRNELVIRVQPNEAVYTKMMTKKPGMFFNPEESELDLTYGNRYKNVKLPDAYERLILDVFCGSQMHFVRSDELREAW"
    "RIFTPLLHQIELEKPKPIPYIYGSRGPTEADELMKRVGFQYEGTYKWVNPHKL"
)


def score(model, tok, device, pairs, apply_mask):
    by_seq = {}
    for seq in {s for p in pairs for s in p}:
        enc = tok([seq], return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
            res = model.encode_protein(enc["input_ids"], enc["attention_mask"])
        by_seq[seq] = res[0, : int(enc["attention_mask"].sum())]

    pad_q = pad_sequence([by_seq[a] for a, _ in pairs], batch_first=True)
    pad_c = pad_sequence([by_seq[b] for _, b in pairs], batch_first=True)
    mask_q = (pad_q.abs().sum(dim=-1) != 0).long()
    mask_c = (pad_c.abs().sum(dim=-1) != 0).long()

    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        logits, valid_mask = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)

    if apply_mask:
        logits = logits.float().masked_fill(~valid_mask, float("-inf"))
        return torch.sigmoid(logits.flatten(1).max(dim=-1).values).cpu().numpy()
    return torch.sigmoid(logits).view(logits.size(0), -1).max(dim=-1).values.cpu().numpy()


def main():
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(device).eval()

    alone = [(SHORT_A, SHORT_B)]
    batched = [(SHORT_A, SHORT_B), (LONG, LONG)]

    fixed = (float(score(model, tok, device, alone, True)[0]),
             float(score(model, tok, device, batched, True)[0]))
    raw = (float(score(model, tok, device, alone, False)[0]),
           float(score(model, tok, device, batched, False)[0]))

    print(f"masked   alone={fixed[0]:.6f} batched={fixed[1]:.6f} "
          f"delta={abs(fixed[1] - fixed[0]):.2e}")
    print(f"unmasked alone={raw[0]:.6f} batched={raw[1]:.6f} "
          f"delta={abs(raw[1] - raw[0]):.2e}")

    assert abs(fixed[1] - fixed[0]) < TOL, (
        f"contact_score changed with batch composition: {fixed[0]} vs {fixed[1]}")
    print("PASS")


if __name__ == "__main__":
    main()
