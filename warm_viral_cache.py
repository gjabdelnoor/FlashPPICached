#!/usr/bin/env python3
"""
One-time pass: encode every viral proteome once and cache it to disk.

Run this BEFORE launching the parallel per-host workers so they never
race to build the same viral cache file (each viral file is used by
all 403 hosts).
"""
import argparse
import glob
import os

import torch
from transformers import AutoModel, AutoTokenizer

from cache_utils import load_or_compute_viral


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viral_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    viral_files = sorted(glob.glob(os.path.join(args.viral_dir, "*.faa")))
    print(f"Warming viral cache for {len(viral_files)} files -> {args.cache_dir}")
    for i, vf in enumerate(viral_files, 1):
        base = os.path.splitext(os.path.basename(vf))[0]
        cache_path = os.path.join(args.cache_dir, base + ".pt")
        if os.path.exists(cache_path):
            print(f"[{i}/{len(viral_files)}] {base} (already cached)")
            continue
        print(f"[{i}/{len(viral_files)}] {base} encoding...")
        load_or_compute_viral(vf, args.cache_dir, model, tokenizer, device, args.batch_size, args.max_len)
    print("Viral cache warm-up complete.")


if __name__ == "__main__":
    main()
