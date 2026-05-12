import os
import torch
import numpy as np
from transformers import AutoTokenizer
from typing import List
from torch.utils.data import Sampler
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_interaction_keys(c1_train, c2_train, c1_val, c2_val, multiplier=1_000_000):
    """
    Builds a global set of interaction keys (ClusterA * M + ClusterB) 
    from both train and validation sets. This serves as the Ground Truth 
    to prevent false negatives during training.
    
    Args:
        c1_train: seq1_cluster array for training set
        c2_train: seq2_cluster array for training set
        c1_val: seq1_cluster array for validation set
        c2_val: seq2_cluster array for validation set
        multiplier: Multiplier for hash key generation (default 1M)
        
    Returns:
        tuple: (valid_keys_tensor, multiplier) where valid_keys_tensor is a 1D tensor of unique interaction keys
    """    
    # Concatenate train and val
    c1 = np.concatenate([c1_train, c1_val])
    c2 = np.concatenate([c2_train, c2_val])
    
    # 2. Filter invalid clusters (-1)
    valid_mask = (c1 != -1) & (c2 != -1)
    c1 = c1[valid_mask]
    c2 = c2[valid_mask]
    
    # 3. Validate cluster IDs don't exceed multiplier (prevent hash collisions)
    max_cluster = max(c1.max(), c2.max()) if len(c1) > 0 else 0
    assert max_cluster < multiplier, (
        f"Cluster ID {max_cluster} exceeds multiplier {multiplier}. "
        f"Hash collisions will occur! Increase multiplier or reduce cluster IDs."
    )
    
    # 4. Create Bidirectional Keys (Vectorized)
    # Key = ClusterA * Multiplier + ClusterB
    keys_fwd = c1 * multiplier + c2
    keys_rev = c2 * multiplier + c1  # Symmetric: If A binds B, B binds A
    
    # 5. Concatenate and Unique
    if len(keys_fwd) == 0:
        print("WARNING: No valid cluster pairs found. All entries have cluster_id=-1.")
        print("False negative masking will be disabled.")
        return torch.empty(0, dtype=torch.long), multiplier
    
    unique_keys = np.unique(np.concatenate([keys_fwd, keys_rev]))
        
    # Return as Tensor and multiplier
    return torch.from_numpy(unique_keys).long(), multiplier


class ClusterSampler(Sampler):
    """
    Sampler that samples examples by individual protein cluster for maximum diversity.
    
    Strategy: 
        1. Sample a protein cluster uniformly at random
        2. Sample an example that contains that cluster (in either seq1 or seq2)
    
    This ensures each protein cluster is equally represented regardless of interaction patterns.
    DDI examples (no cluster, cluster=-1) are sampled according to ddi_sampling_prob.
    Compatible with Accelerate/DDP via set_epoch() method.
    """
    def __init__(self, c1: np.ndarray, c2: np.ndarray, ddi_sampling_prob: float = 0.5, seed: int = 42):
        """
        Args:
            c1: seq1_cluster array (numpy array of int64, -1 for DDI/missing)
            c2: seq2_cluster array (numpy array of int64, -1 for DDI/missing)
            ddi_sampling_prob: Probability of sampling DDI examples (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_samples = len(c1)
        self.ddi_sampling_prob = ddi_sampling_prob
        self.seed = seed
        self.epoch = 0
        
        # Build cluster index mapping: each cluster maps to examples containing it
        self.cluster_to_indices = defaultdict(set)  # Use set to avoid duplicates
        self.ddi_indices = []
        
        for idx in range(len(c1)):
            if c1[idx] == -1 or c2[idx] == -1:
                self.ddi_indices.append(idx)
            else:
                # Add example to both cluster indices
                self.cluster_to_indices[int(c1[idx])].add(idx)
                self.cluster_to_indices[int(c2[idx])].add(idx)
        
        # Precompute cluster list and convert indices to numpy arrays for efficiency
        self.clusters = list(self.cluster_to_indices.keys())
        self.cluster_to_indices = {k: np.array(list(v), dtype=np.int64) for k, v in self.cluster_to_indices.items()}
        self.ddi_indices = np.array(self.ddi_indices, dtype=np.int64) if self.ddi_indices else np.array([], dtype=np.int64)
        
        self.has_ddi = len(self.ddi_indices) > 0
        self.has_clusters = len(self.clusters) > 0
    
    def __iter__(self):
        # Use torch generator for better integration with PyTorch/DDP
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = []
        for _ in range(self.num_samples):
            # Decide whether to sample from DDI or clusters
            if self.has_ddi and (not self.has_clusters or torch.rand(1, generator=g).item() < self.ddi_sampling_prob):
                # Sample from DDI
                idx = self.ddi_indices[torch.randint(len(self.ddi_indices), (1,), generator=g).item()]
            elif self.has_clusters:
                # Sample a cluster uniformly, then sample an example from it
                cluster_idx = torch.randint(len(self.clusters), (1,), generator=g).item()
                cluster = self.clusters[cluster_idx]
                cluster_examples = self.cluster_to_indices[cluster]
                idx = cluster_examples[torch.randint(len(cluster_examples), (1,), generator=g).item()]
            else:
                # Fallback: sample uniformly (shouldn't happen in practice)
                idx = torch.randint(self.num_samples, (1,), generator=g).item()
            
            indices.append(int(idx))
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        """Set epoch for DDP training to ensure different shuffling per epoch."""
        self.epoch = epoch


def create_contact_targets(
    seq1_lens: List[int],
    seq2_lens: List[int],
    contacts_list: List[List],
    target_map_size: int
) -> torch.Tensor:
    """
    Create contact target tensor from list of contacts.
    
    Args:
        seq1_lens: List of sequence 1 lengths
        seq2_lens: List of sequence 2 lengths
        contacts_list: List of contact lists, each contact is [i, j]
        target_map_size: Size of the target map
        
    Returns:
        contact_targets: (batch_size, target_map_size, target_map_size) bool tensor
    """
    batch_size = len(seq1_lens)
    contact_targets = torch.zeros(batch_size, target_map_size, target_map_size, dtype=torch.bool)
    
    batch_indices_list = []
    r1_indices_list = []
    r2_indices_list = []
    
    for i, (seq1_len, seq2_len, contacts) in enumerate(zip(seq1_lens, seq2_lens, contacts_list)):
        seq1_len = min(seq1_len, target_map_size)
        seq2_len = min(seq2_len, target_map_size)
        
        if contacts:
            contacts_tensor = torch.tensor(contacts, dtype=torch.long)
            valid_mask = (contacts_tensor[:, 0] < seq1_len) & (contacts_tensor[:, 1] < seq2_len)
            valid_contacts = contacts_tensor[valid_mask]
            
            if len(valid_contacts) > 0:
                batch_indices_list.append(torch.full((len(valid_contacts),), i, dtype=torch.long))
                r1_indices_list.append(valid_contacts[:, 0])
                r2_indices_list.append(valid_contacts[:, 1])
    
    if batch_indices_list:
        batch_indices = torch.cat(batch_indices_list)
        r1_indices = torch.cat(r1_indices_list)
        r2_indices = torch.cat(r2_indices_list)
        contact_targets[batch_indices, r1_indices, r2_indices] = True
    
    return contact_targets


class DataCollatorForPPI:
    """
    Custom data collator.
    - Tokenizes seq1 and seq2 separately.
    - Creates the 2D contact map target tensor.
    - Optionally swaps query/target with 50% probability.
    """
    def __init__(self, tokenizer: AutoTokenizer, max_len: int, swap_augment: bool = True, adds_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.swap_augment = swap_augment
        self.adds_special_tokens = adds_special_tokens  # ESM adds CLS/SEP, gLM2 does not

    def __call__(self, batch: list[dict]):
        # Randomly swap query/target for augmentation
        if self.swap_augment:
            batch = [
                {'seq1': item['seq2'],
                 'seq2': item['seq1'], 
                 'contacts': [(j, i) for i, j in item.get('contacts', [])],
                 'seq1_cluster': item.get('seq2_cluster'),
                 'seq2_cluster': item.get('seq1_cluster'),
                 'complex_id': item.get('complex_id')}
                if torch.rand(1).item() < 0.5 else item
                for item in batch
            ]
        
        # Extract sequences
        seqs1 = [item['seq1'] for item in batch]
        seqs2 = [item['seq2'] for item in batch]
        
        # Normalize cluster IDs -1 (use -1 for missing/DDI dataset)
        seq1_clusters = [item.get('seq1_cluster') for item in batch]
        seq1_clusters = [-1 if val is None else val for val in seq1_clusters]
        seq2_clusters = [item.get('seq2_cluster') for item in batch]
        seq2_clusters = [-1 if val is None else val for val in seq2_clusters]
        
        # Tokenize separately (ESM adds CLS/SEP, gLM2 does not)
        batch1_tokens = self.tokenizer(
            seqs1, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len
        )
        
        batch2_tokens = self.tokenizer(
            seqs2, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len
        )
        
        # Create contact targets using standalone function
        seq1_lens = [len(item['seq1']) for item in batch]
        seq2_lens = [len(item['seq2']) for item in batch]
        # The target map size matches model output (which slices off CLS/SEP if present)
        target_map_size = self.max_len - 2 if self.adds_special_tokens else self.max_len
        contacts_list = [item.get('contacts', []) for item in batch]
        contact_targets = create_contact_targets(seq1_lens, seq2_lens, contacts_list, target_map_size)

        return {
            'seq1_tokens': batch1_tokens,
            'seq2_tokens': batch2_tokens,
            'contact_target': contact_targets,
            'seq1_cluster': torch.tensor(seq1_clusters, dtype=torch.long),
            'seq2_cluster': torch.tensor(seq2_clusters, dtype=torch.long),
        }


