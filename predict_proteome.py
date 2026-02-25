import argparse
import torch
import faiss
import numpy as np
import pandas as pd
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

def main():
    parser = argparse.ArgumentParser(description="FlashPPI: Fast Proteome-wide PPI Screening")
    parser.add_argument("--fasta", type=str, required=True, help="Path to the input FASTA file containing the proteome.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save the output CSV. Default: predictions.csv")
    parser.add_argument("--model_name", type=str, default="tattabio/flashppi", help="HuggingFace model ID or local path.")
    parser.add_argument("--stage1_top_k", type=int, default=100, help="Number of nearest neighbors to retrieve per protein in stage 1.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Contact score threshold to keep predictions.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for model inference.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length.")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model and tokenizer ({args.model_name}) to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device).eval()

    print(f"Loading sequences from {args.fasta}...")
    records = list(SeqIO.parse(args.fasta, "fasta"))
    sequences = [str(r.seq) for r in records]
    ids = [r.id for r in records]

    if len(sequences) < 2:
        raise ValueError("FASTA file must contain at least 2 sequences to predict interactions.")

    print("Stage 1: Encoding proteins...")
    query_embeds, key_embeds, residue_embeds_list, seq_lengths = [], [], [], []

    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        for i in range(0, len(sequences), args.batch_size):
            batch_seqs = sequences[i : i + args.batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len).to(device)
            
            res_embed = model.encode_protein(inputs["input_ids"], inputs["attention_mask"])
            query_embeds.append(model.head_q(res_embed, inputs["attention_mask"]).cpu().float().numpy())
            key_embeds.append(model.head_k(res_embed, inputs["attention_mask"]).cpu().float().numpy())
            
            lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for j, seq_len in enumerate(lengths):
                residue_embeds_list.append(res_embed[j, :int(seq_len), :])
                seq_lengths.append(int(seq_len))

    query_embeds = np.concatenate(query_embeds, axis=0)
    key_embeds = np.concatenate(key_embeds, axis=0)
    residue_embeds = torch.cat(residue_embeds_list, dim=0)
    
    cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.long, device=device)
    cu_seqlens[1:] = torch.tensor(seq_lengths, dtype=torch.long, device=device).cumsum(0)

    print("Stage 1.5: FAISS Retrieval...")
    index = faiss.IndexFlatIP(key_embeds.shape[1])
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(key_embeds)
    
    search_k = min(args.stage1_top_k + 1, len(sequences))
    _, I = index.search(query_embeds, k=search_k)

    inference_tasks = [(q, int(c)) for q in range(len(sequences)) for c in I[q] if c >= 0 and c != q]
    print(f"Found {len(inference_tasks)} candidate pairs. Starting fine-grained contact prediction...")

    results = []
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
        for i in range(0, len(inference_tasks), args.batch_size):
            batch = inference_tasks[i : i + args.batch_size]
            q_batch = [residue_embeds[cu_seqlens[q]:cu_seqlens[q+1]] for q, _ in batch]
            c_batch = [residue_embeds[cu_seqlens[c]:cu_seqlens[c+1]] for _, c in batch]
            
            pad_q = pad_sequence(q_batch, batch_first=True)
            pad_c = pad_sequence(c_batch, batch_first=True)
            mask_q = (pad_q.abs().sum(dim=-1) != 0).long()
            mask_c = (pad_c.abs().sum(dim=-1) != 0).long()
            
            logits, _ = model.predict_contacts(pad_q, pad_c, mask_q, mask_c)
            scores = torch.sigmoid(logits).view(logits.size(0), -1).max(dim=-1).values.cpu().numpy()
            
            for k, (q_idx, c_idx) in enumerate(batch):
                if scores[k] > args.threshold:
                    results.append({
                        "query_id": ids[q_idx], 
                        "match_id": ids[c_idx], 
                        "contact_score": float(scores[k])
                    })

    # Deduplicate and save
    df = pd.DataFrame(results)
    if not df.empty:
        # Sort so pair keys are consistent for deduplication
        df['pair_key'] = df.apply(lambda r: tuple(sorted([r['query_id'], r['match_id']])), axis=1)
        # Keep the maximum score for each unique pair
        df = df.loc[df.groupby('pair_key')['contact_score'].idxmax()].drop(columns=['pair_key'])
        df = df.sort_values(by='contact_score', ascending=False)
        
        df.to_csv(args.output, index=False)
        print(f"Success! Saved {len(df)} high-confidence predictions to {args.output}")
    else:
        print(f"No interactions found above the threshold of {args.threshold}.")

if __name__ == "__main__":
    main()