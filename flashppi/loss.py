import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


def focal_loss(logits, targets, valid_mask, gamma=2.0, alpha=0.25):
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
   
    logits_valid = logits[valid_mask]
    targets_valid = targets[valid_mask]
    
    bce_loss = F.binary_cross_entropy_with_logits(logits_valid, targets_valid, reduction='none')
    pt = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + targets_valid * (2 * alpha - 1)
    f_loss = alpha_tensor * (1 - pt) ** gamma * bce_loss
    
    num_positives = targets_valid.sum()
    if num_positives == 0:
        return f_loss.sum()
    else:
        return f_loss.sum() / num_positives

class ContactLoss(nn.Module):
    def __init__(self, use_focal_loss=False):
        super().__init__()
        self.use_focal_loss = use_focal_loss

    def forward(self, contact_logits, contact_target, valid_mask):
        """
        Args:
            contact_logits: (B', L1, L2) contact prediction logits
            contact_target: (B, L1, L2) boolean contact targets
            valid_mask: (B', L1, L2) boolean mask, True for valid positions, False for padding
        """
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=contact_logits.device, requires_grad=True)
        
        # Expand contact_target if contact_logits has more pairs (negatives)
        if contact_logits.shape[0] > contact_target.shape[0]:
            num_neg = contact_logits.shape[0] - contact_target.shape[0]
            neg_target = torch.zeros(num_neg, *contact_target.shape[1:], 
                                     device=contact_target.device, dtype=contact_target.dtype)
            contact_target = torch.cat([contact_target, neg_target], dim=0)

        # Cast boolean targets to float for loss computation
        if self.use_focal_loss:
            loss = focal_loss(contact_logits, contact_target.float(), valid_mask, gamma=2.0, alpha=0.25)
        else:
            # Only compute loss on valid positions
            loss = F.binary_cross_entropy_with_logits(
                contact_logits[valid_mask], 
                contact_target.float()[valid_mask],
                reduction='mean'
            )
        
        return loss

class ClipLoss(nn.Module):
    """Standard InfoNCE for Evaluation (No GradCache logic)."""
    def __init__(self, local_loss=False, gather_with_grad=False, rank=0, world_size=1, valid_keys_tensor=None, multiplier=1_000_000):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.multiplier = multiplier
        # Register keys as buffer so they automatically move to GPU with .to(device)
        if valid_keys_tensor is not None:
            self.register_buffer('valid_keys', valid_keys_tensor)
        else:
            self.register_buffer('valid_keys', None)

    def forward(self, features1, features2, logit_scale, seq1_clusters=None, seq2_clusters=None, compute_top1_acc=False):
        device = features1.device

        # Gather for global view if distributed
        if self.world_size > 1:
            if self.gather_with_grad:
                all_f1 = torch.cat(torch.distributed.nn.all_gather(features1), dim=0)
                all_f2 = torch.cat(torch.distributed.nn.all_gather(features2), dim=0)
                if seq1_clusters is not None:
                    all_seq1_clusters = torch.cat(torch.distributed.nn.all_gather(seq1_clusters), dim=0)
                    all_seq2_clusters = torch.cat(torch.distributed.nn.all_gather(seq2_clusters), dim=0)
                else:
                    all_seq1_clusters = None
                    all_seq2_clusters = None
            else:
                gathered1 = [torch.zeros_like(features1) for _ in range(self.world_size)]
                gathered2 = [torch.zeros_like(features2) for _ in range(self.world_size)]
                dist.all_gather(gathered1, features1)
                dist.all_gather(gathered2, features2)
                all_f1 = torch.cat(gathered1, dim=0)
                all_f2 = torch.cat(gathered2, dim=0)

                if seq1_clusters is not None:
                    gathered_c1 = [torch.zeros_like(seq1_clusters) for _ in range(self.world_size)]
                    gathered_c2 = [torch.zeros_like(seq2_clusters) for _ in range(self.world_size)]
                    dist.all_gather(gathered_c1, seq1_clusters)
                    dist.all_gather(gathered_c2, seq2_clusters)
                    all_seq1_clusters = torch.cat(gathered_c1, dim=0)
                    all_seq2_clusters = torch.cat(gathered_c2, dim=0)
                else:
                    all_seq1_clusters = None
                    all_seq2_clusters = None
        else:
            all_f1, all_f2 = features1, features2
            all_seq1_clusters = seq1_clusters
            all_seq2_clusters = seq2_clusters

        logits = logit_scale * all_f1 @ all_f2.T
        logits = mask_false_negatives(logits, all_seq1_clusters, all_seq2_clusters, self.valid_keys, self.multiplier)
        labels = torch.arange(len(logits), device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        if compute_top1_acc:
            with torch.no_grad():
                top1_acc = (logits.argmax(dim=1) == labels).float().mean().item()
            return loss, top1_acc
        return loss


def mask_false_negatives(logits, seq1_clusters, seq2_clusters, valid_keys, multiplier=1_000_000):
    """
    Shared helper to mask false negatives based on cluster interactions.
    
    Args:
        logits: (B, B) similarity matrix
        seq1_clusters: (B,) cluster IDs for sequence 1
        seq2_clusters: (B,) cluster IDs for sequence 2
        valid_keys: Tensor of valid interaction keys
        multiplier: Multiplier for generating unique keys
    
    Returns:
        logits: (B, B) masked similarity matrix
    """
    if seq1_clusters is None or seq2_clusters is None or valid_keys is None or len(valid_keys) == 0:
        return logits
    
    # Generate keys for the batch: (B, 1) * M + (1, B) -> (B, B) keys
    batch_keys = seq1_clusters.unsqueeze(1) * multiplier + seq2_clusters.unsqueeze(0)
    
    # Ensure valid_keys is on the same device as batch_keys
    valid_keys = valid_keys.to(batch_keys.device)
    
    mask = torch.isin(batch_keys, valid_keys)
    
    mask.fill_diagonal_(False)
    
    # Apply Mask
    logits = logits.masked_fill(mask, -1e4)
    return logits


class GradCacheClipLoss(nn.Module):
    """Helper Loss for the Global Cache Step only."""
    def __init__(self, rank=0, world_size=1, valid_keys_tensor=None, multiplier=1_000_000):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.multiplier = multiplier
        # Register keys as buffer so they automatically move to GPU with .to(device)
        if valid_keys_tensor is not None:
            self.register_buffer('valid_keys', valid_keys_tensor)
        else:
            self.register_buffer('valid_keys', None)

    def forward(self, local_embed1, local_embed2, logit_scale, seq1_clusters=None, seq2_clusters=None, compute_top1_acc=False):
        # We receive local embeddings (with gradient enabled at the leaf).
        # We must gather the rest detached.
        device = local_embed1.device
        
        if self.world_size > 1:
            # Gather detached versions from everyone
            all_e1 = [torch.zeros_like(local_embed1) for _ in range(self.world_size)]
            all_e2 = [torch.zeros_like(local_embed2) for _ in range(self.world_size)]
            
            dist.all_gather(all_e1, local_embed1.detach())
            dist.all_gather(all_e2, local_embed2.detach())
            
            # Swap in our local version that preserves the computation graph
            all_e1[self.rank] = local_embed1
            all_e2[self.rank] = local_embed2
            
            global_e1 = torch.cat(all_e1, dim=0)
            global_e2 = torch.cat(all_e2, dim=0)
            
            # Gather cluster IDs if provided
            if seq1_clusters is not None:
                all_c1 = [torch.zeros_like(seq1_clusters) for _ in range(self.world_size)]
                all_c2 = [torch.zeros_like(seq2_clusters) for _ in range(self.world_size)]
                dist.all_gather(all_c1, seq1_clusters)
                dist.all_gather(all_c2, seq2_clusters)
                global_seq1_clusters = torch.cat(all_c1, dim=0)
                global_seq2_clusters = torch.cat(all_c2, dim=0)
            else:
                global_seq1_clusters = None
                global_seq2_clusters = None
        else:
            global_e1 = local_embed1
            global_e2 = local_embed2
            global_seq1_clusters = seq1_clusters
            global_seq2_clusters = seq2_clusters

        logits = logit_scale * global_e1 @ global_e2.T
        
        # Mask false negatives using shared helper
        logits = mask_false_negatives(logits, global_seq1_clusters, global_seq2_clusters, self.valid_keys, self.multiplier)
        
        labels = torch.arange(len(logits), device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        
        if compute_top1_acc:
            with torch.no_grad():
                top1_pred = logits.argmax(dim=1)
                correct_indices = torch.arange(len(logits), device=device)
                top1_acc = (top1_pred == correct_indices).float().mean().item()
            return loss, top1_acc
        return loss


class PPIGradientCache:
    """
    Gradient Caching for FlashPPI.
    """
    def __init__(self, model, chunk_size, clip_loss_fn, contact_loss_fn, contact_weight, accelerator, clip_weight=1.0):
        self.model = model
        self.chunk_size = chunk_size
        self.clip_loss_fn = clip_loss_fn
        self.contact_loss_fn = contact_loss_fn
        self.contact_weight = contact_weight
        self.clip_weight = clip_weight
        self.accelerator = accelerator
        self.device = accelerator.device

    def split_data(self, batch):
        seq1 = batch['seq1_tokens']
        seq2 = batch['seq2_tokens']
        targets = batch['contact_target']
        seq1_clusters = batch.get('seq1_cluster', None)
        seq2_clusters = batch.get('seq2_cluster', None)
        
        batch_size = targets.shape[0]
        chunks = []
        
        for i in range(0, batch_size, self.chunk_size):
            chunk = {
                's1_ids': seq1['input_ids'][i:i+self.chunk_size].to(self.device),
                's1_mask': seq1['attention_mask'][i:i+self.chunk_size].to(self.device),
                's2_ids': seq2['input_ids'][i:i+self.chunk_size].to(self.device),
                's2_mask': seq2['attention_mask'][i:i+self.chunk_size].to(self.device),
                'targets': targets[i:i+self.chunk_size].to(self.device),
            }
            if seq1_clusters is not None:
                chunk['seq1_cluster'] = seq1_clusters[i:i+self.chunk_size].to(self.device)
            if seq2_clusters is not None:
                chunk['seq2_cluster'] = seq2_clusters[i:i+self.chunk_size].to(self.device)
            chunks.append(chunk)
        return chunks, seq1_clusters, seq2_clusters

    def set_random_state(self, state):
        cpu_rng, gpu_rng = state
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng, self.device)

    def step(self, batch):
        chunks, seq1_clusters, seq2_clusters = self.split_data(batch)
        
        # --- PASS 1: CACHE (No Grad) ---
        local_e1_cache = []
        local_e2_cache = []
        rnd_states = []
        
        unwrapped_model = self.model.module if hasattr(self.model, "module") else self.model
        # Initialize defaults
        clip_loss_val = 0.0
        top1_acc = 0.0

        local_clusters1_cache = []
        local_clusters2_cache = []
        with torch.no_grad(), self.accelerator.autocast():
            for chunk in chunks:
                rnd_states.append((torch.get_rng_state(), torch.cuda.get_rng_state(self.device)))
                e1, e2, clusters1_chunk, clusters2_chunk = unwrapped_model.embed_pair(
                    chunk['s1_ids'], chunk['s2_ids'],
                    chunk['s1_mask'], chunk['s2_mask'],
                    chunk.get('seq1_cluster'), chunk.get('seq2_cluster')
                )
                local_e1_cache.append(e1)
                local_e2_cache.append(e2)
                if clusters1_chunk is not None:
                    local_clusters1_cache.append(clusters1_chunk)
                if clusters2_chunk is not None:
                    local_clusters2_cache.append(clusters2_chunk)

        big_e1 = torch.cat(local_e1_cache, dim=0).detach().requires_grad_()
        big_e2 = torch.cat(local_e2_cache, dim=0).detach().requires_grad_()

        if local_clusters1_cache and local_clusters2_cache:
            seq1_clusters = torch.cat(local_clusters1_cache, dim=0)
            seq2_clusters = torch.cat(local_clusters2_cache, dim=0)
        else:
            seq1_clusters = None
            seq2_clusters = None

        logit_scale = unwrapped_model.logit_scale.exp()
        clip_loss, top1_acc = self.clip_loss_fn(big_e1, big_e2, logit_scale, seq1_clusters=seq1_clusters, seq2_clusters=seq2_clusters, compute_top1_acc=True)
        clip_loss_val = clip_loss.item()
        self.accelerator.backward(clip_loss)

        if unwrapped_model.logit_scale.grad is not None:
            saved_scale_grad = unwrapped_model.logit_scale.grad.clone()
            unwrapped_model.logit_scale.grad.zero_()
        else:
            saved_scale_grad = None

        world_size = self.accelerator.num_processes
        grads1 = big_e1.grad * world_size
        grads2 = big_e2.grad * world_size
        effective_chunk_size = self.chunk_size * 2 if swap_self_negative else self.chunk_size
        chunked_grads1 = torch.split(grads1, effective_chunk_size)
        chunked_grads2 = torch.split(grads2, effective_chunk_size)

        # --- PASS 2: GRADIENT STEP (With Grad) ---
        total_contact_loss = 0.0
        logit_scale_val = None
        
        for i, chunk in enumerate(chunks):
            self.set_random_state(rnd_states[i])
            is_last_chunk = (i == len(chunks) - 1)
            context = nullcontext() if is_last_chunk else self.accelerator.no_sync(self.model)
            
            with context:
                outputs = self.model(
                    chunk['s1_ids'], chunk['s2_ids'], 
                    chunk['s1_mask'], chunk['s2_mask'],
                    chunk.get('seq1_cluster'), chunk.get('seq2_cluster')
                )
                c1, c2, c_logits, logit_scale_out, c_valid_mask = outputs[:5]
                
                # Capture logit_scale from the last chunk for logging
                if is_last_chunk:
                    logit_scale_val = logit_scale_out.item()
                
                surrogate = (torch.sum(c1 * chunked_grads1[i]) + torch.sum(c2 * chunked_grads2[i])) * self.clip_weight
                
                # Contact Loss
                c_loss = torch.tensor(0.0, device=self.device)
                if c_logits is not None:
                    c_loss = self.contact_loss_fn(c_logits, chunk['targets'], c_valid_mask)
                    c_loss = c_loss / len(chunks)
                    total_contact_loss += c_loss.item()
                
                # DDP expects logit_scale to be part of the graph in the backward pass of the syncing step.
                dummy_loss = logit_scale_out * 0.0
                if c_logits is not None:
                     dummy_loss = dummy_loss + (c_logits.sum() * 0.0)

                loss_to_backward = surrogate + c_loss * self.contact_weight + dummy_loss

                self.accelerator.backward(loss_to_backward)
        
        if saved_scale_grad is not None:
            if unwrapped_model.logit_scale.grad is None:
                unwrapped_model.logit_scale.grad = saved_scale_grad
            else:
                unwrapped_model.logit_scale.grad += saved_scale_grad
            
        return clip_loss_val, total_contact_loss, logit_scale_val, top1_acc
