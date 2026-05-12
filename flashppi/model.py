import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional
from transformers import AutoModel
from .config import ModelConfig
from .layers import TransformerConfig, MLPHead, TransformerLayers


def load_plm_model(model_name: str):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    # Remove unused ESM modules that cause DDP issues
    if hasattr(model, 'pooler'):
        model.pooler = None
    if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'position_embeddings'):
        model.embeddings.position_embeddings = None
    return model


def freeze_plm_layers(model: nn.Module, num_layers: int):
    if num_layers <= 0:
        return
    
    # Try to find the encoder layers
    layers = None
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        layers = model.encoder.layer
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        layers = model.encoder.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    
    if layers is None:
        print(f"WARNING: Could not find encoder layers in model. Skipping layer freezing.")
        return
    
    num_frozen = 0
    for layer in layers[:num_layers]:
        layer.requires_grad_(False)
        num_frozen += 1

    # Freeze embeddings
    if hasattr(model, "embeddings"):
        model.embeddings.requires_grad_(False)
    if hasattr(model, "tok_embeddings"):
        model.tok_embeddings.requires_grad_(False)
    
    print(f"Frozen {num_frozen} out of {len(layers)} PLM layers.")


def get_contact_negatives(
    embed1: torch.Tensor, 
    embed2: torch.Tensor, 
    mask1: torch.Tensor, 
    mask2: torch.Tensor, 
    random_neg_ratio: float = 1.0,
    self_neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.0,
    clusters1: Optional[torch.Tensor] = None,
    clusters2: Optional[torch.Tensor] = None,
    valid_keys: Optional[torch.Tensor] = None,
    multiplier: int = 1_000_000,
    clip_embed1: Optional[torch.Tensor] = None,
    clip_embed2: Optional[torch.Tensor] = None,
):
    """
    Generate negative pairs for contact prediction, avoiding known interactions.
    Supports random, self, and hard negatives (based on CLIP similarity).
    """
    B = embed1.size(0)
    device = embed1.device
    
    # Pre-compute known interaction mask (B, B) - True means it's a known interaction
    known_mask = None
    if clusters1 is not None and clusters2 is not None and valid_keys is not None and len(valid_keys) > 0:
        batch_keys = clusters1.unsqueeze(1) * multiplier + clusters2.unsqueeze(0)
        invalid = (clusters1.unsqueeze(1) == -1) | (clusters2.unsqueeze(0) == -1)
        known_mask = torch.isin(batch_keys, valid_keys.to(device)) & ~invalid
    
    # Collect negative pair indices
    neg_idx1_list, neg_idx2_list = [], []
    
    # Random negatives: shift pairs (A_i, B_{i+shift}), skip known interactions
    if random_neg_ratio > 0 and B > 1:
        num_rand = int(B * random_neg_ratio)
        reps = (num_rand + B - 1) // B
        for shift in range(1, reps + 1):
            n = min(B, num_rand - (shift - 1) * B)
            idx1 = torch.arange(n, device=device)
            idx2 = (idx1 + shift) % B
            if known_mask is not None:
                valid = ~known_mask[idx1, idx2]
                idx1, idx2 = idx1[valid], idx2[valid]
            neg_idx1_list.append(idx1)
            neg_idx2_list.append(idx2)
    
    # Self negatives: (A_i, A_i), skip known interactions
    if self_neg_ratio > 0:
        n = min(B, int(B * self_neg_ratio))
        idx = torch.arange(n, device=device)
        if known_mask is not None:
            valid = ~known_mask[idx, idx]
            idx = idx[valid]
        neg_idx1_list.append(idx)
        neg_idx2_list.append(idx)
    
    # Hard negatives: select most similar non-known pairs by CLIP score
    if hard_neg_ratio > 0 and clip_embed1 is not None and clip_embed2 is not None and B > 1:
        num_hard_per_example = max(1, int(hard_neg_ratio))
        with torch.no_grad():
            sim_matrix = clip_embed1 @ clip_embed2.T
            # Mask out diagonal (positives)
            sim_matrix.fill_diagonal_(-1e9)
            # Mask out known interactions before selecting top-K
            if known_mask is not None:
                sim_matrix = sim_matrix.masked_fill(known_mask, -1e9)
            # Get top-K most similar valid pairs per example
            num_hard_actual = min(num_hard_per_example, B - 1)
            _, top_indices_per_row = torch.topk(sim_matrix, num_hard_actual, dim=1)
            idx1 = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_hard_actual).flatten()
            idx2 = top_indices_per_row.flatten()
        neg_idx1_list.append(idx1)
        neg_idx2_list.append(idx2)
    
    # Combine all negatives
    if neg_idx1_list:
        neg_idx1 = torch.cat(neg_idx1_list)
        neg_idx2 = torch.cat(neg_idx2_list)
        
        if len(neg_idx1) > 0:
            return embed1[neg_idx1], embed2[neg_idx2], mask1[neg_idx1], mask2[neg_idx2]
    
    # Return empty if no negatives
    return (torch.empty(0, *embed1.shape[1:], device=device),
            torch.empty(0, *embed2.shape[1:], device=device),
            torch.empty(0, *mask1.shape[1:], device=device),
            torch.empty(0, *mask2.shape[1:], device=device))


class ContactHead(nn.Module):
    """
    Computes interactions between two proteins using Multi-Head Attention logic.
    Returns the raw attention logits (the contact map) for supervision.
    """
    def __init__(self, input_dim, contact_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = contact_dim // num_heads
        assert contact_dim % num_heads == 0, "Input dim must be divisible by num_heads"


        # Segment Embedding
        self.segment_embed = nn.Embedding(2, input_dim)
        nn.init.normal_(self.segment_embed.weight, std=0.02)
        
        self.transformer = TransformerLayers(TransformerConfig(dim=input_dim, heads=num_heads, depth=2, out_dim=input_dim))

        # Layer normalization before projection (stabilizes training)
        self.norm = nn.LayerNorm(input_dim)

        # Projections for Query (Seq1) and Key (Seq2)
        self.q_proj = nn.Linear(input_dim, contact_dim)
        self.k_proj = nn.Linear(input_dim, contact_dim)
        
        self.output_mix = nn.Linear(num_heads, 1)
        nn.init.constant_(self.output_mix.bias, -3.0)

        # Scale factor (1 / sqrt(head_dim))
        self.scale = self.head_dim ** -0.5

    def forward(self, embed1, embed2, mask1, mask2):
        B, L1, D = embed1.shape
        _, L2, _ = embed2.shape

        seg1 = self.segment_embed(torch.zeros(L1, device=embed1.device, dtype=torch.long))
        seg2 = self.segment_embed(torch.ones(L2, device=embed1.device, dtype=torch.long))
        
        # Concat inputs
        x = torch.cat([embed1 + seg1.unsqueeze(0), embed2 + seg2.unsqueeze(0)], dim=1)
        
        combined_mask = None
        if mask1 is not None and mask2 is not None:
            combined_mask = torch.cat([mask1, mask2], dim=1).bool()
            
        # Run Transformer
        x = self.transformer(x, attention_mask=combined_mask)
        
        # Split & Bottleneck
        embed1 = x[:, :L1, :]
        embed2 = x[:, L1:, :]

        embed1 = self.norm(embed1)
        embed2 = self.norm(embed2)

        q = self.q_proj(embed1).view(B, L1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(embed2).view(B, L2, self.num_heads, self.head_dim).transpose(1, 2)

        # Shape: (Batch, Heads, L1, L2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Permute to (Batch, L1, L2, Heads) to apply the mixing Linear layer
        attn_logits = attn_logits.permute(0, 2, 3, 1).contiguous()
        
        # Shape: (Batch, L1, L2, 1) -> (Batch, L1, L2)
        contact_logits = self.output_mix(attn_logits).squeeze(-1)

        if mask1 is not None and mask2 is not None:
             # Shape: (Batch, L1, L2)
            valid_mask = (mask1.unsqueeze(2) * mask2.unsqueeze(1)).bool()
        else:
            valid_mask = torch.ones_like(contact_logits, dtype=torch.bool)

        return contact_logits, valid_mask


class PoolLayer(nn.Module):
    def __init__(self, method: Literal["mean", "max"] = "mean") -> None:
        super().__init__()
        self.method = method
    
    def forward(self, embeds: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        mask = mask.unsqueeze(-1).bool()
        if self.method == "mean":
            embeds = torch.where(mask, embeds, 0.0)
            embeds = torch.sum(embeds, -2)
            embeds /= torch.clamp(torch.sum(mask, dim=-2, dtype=embeds.dtype), min=1.0)
        elif self.method == "max":
            embeds = torch.where(mask, embeds, torch.finfo(embeds.dtype).min)
            embeds = torch.max(embeds, dim=-2).values
            embeds = torch.where(mask.any(-2), embeds, torch.zeros_like(embeds))
        return embeds

class ContrastiveHead(nn.Module):
    """
    Linear projection, pooling, and MLP head, normalization for contrastive learning.
    """
    def __init__(self, hidden_dim, clip_embed_dim):
        super().__init__()
        self.pool_layer = PoolLayer(method="mean")
        self.head = MLPHead(hidden_dim, clip_embed_dim)

    def forward(self, residue_embeds, mask):
        pool_embeds = self.pool_layer(residue_embeds, mask)
        return F.normalize(self.head(pool_embeds), dim=-1)

class FlashPPIModel(nn.Module):
    def __init__(
        self, 
        config: ModelConfig,
        swap_self_negative: bool = False,
        contact_random_neg_ratio: float = 1.0,
        contact_self_neg_ratio: float = 1.0,
        contact_hard_neg_ratio: float = 0.0,
        valid_keys_tensor: Optional[torch.Tensor] = None,
        multiplier: int = 1_000_000,
    ):
        super().__init__()
        self.config = config
        self.swap_self_negative = swap_self_negative
        self.contact_random_neg_ratio = contact_random_neg_ratio
        self.contact_self_neg_ratio = contact_self_neg_ratio
        self.contact_hard_neg_ratio = contact_hard_neg_ratio
        # Register valid_keys as buffer for contact negative filtering
        if valid_keys_tensor is not None:
            self.multiplier = multiplier
            self.register_buffer('valid_keys', valid_keys_tensor, persistent=False)

        self.plm = load_plm_model(config.plm_model_name)
        
        # Freeze logic
        if config.num_frozen_layers > 0:
            freeze_plm_layers(self.plm, config.num_frozen_layers)

        if hasattr(self.plm.config, 'hidden_size'):
            hidden_dim = self.plm.config.hidden_size
        else:
            hidden_dim = self.plm.config.dim
        
        # ESM tokenizer adds cls/eos tokens.
        self._adds_special_tokens = "esm" in config.plm_model_name.lower()
        self.head_q = ContrastiveHead(hidden_dim, config.clip_embed_dim)
        self.head_k = ContrastiveHead(hidden_dim, config.clip_embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Contact Head
        self.contact_head = ContactHead(hidden_dim, config.contact_embed_dim, num_heads=8)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the PLM backbone and contact head transformer if supported."""
        if hasattr(self.plm, 'gradient_checkpointing_enable'):
            self.plm.gradient_checkpointing_enable()
        self.contact_head.transformer.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the PLM backbone and contact head transformer if supported."""
        if hasattr(self.plm, 'gradient_checkpointing_disable'):
            self.plm.gradient_checkpointing_disable()
        self.contact_head.transformer.gradient_checkpointing = False


    def encode_protein(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Runs the shared plm model on tokenized protein sequences.
        
        Args:
            input_ids (torch.Tensor): (B, L) - Token IDs from the tokenizer
            attention_mask (Optional[torch.Tensor]): (B, L) - Attention mask (1 for real tokens, 0 for padding)
        
        Returns:
            residue_embeds (torch.Tensor): (B, L, D_plm) - Embeddings for each residue
            pool_embed (torch.Tensor): (B, D_plm) - Pooled embedding using mask-aware pooling
        """
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        
        # (B, L, D_plm) - Full sequence embeddings
        residue_embeds = outputs.last_hidden_state
        return residue_embeds

    def predict_contacts(self, embed1: torch.Tensor, embed2: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
        """
        Predicts a contact map using dot product between projected residue embeddings.
        
        Args:
            embed1 (torch.Tensor): (B, L1, D_plm) - Residue embeddings for sequence 1 (Query)
            embed2 (torch.Tensor): (B, L2, D_plm) - Residue embeddings for sequence 2 (Key)
            mask1 (torch.Tensor): (B, L1) - Attention mask for seq1
            mask2 (torch.Tensor): (B, L2) - Attention mask for seq2
        
        Returns:
            contact_logits (torch.Tensor): (B, L1, L2) - Contact prediction logits
        """
        return self.contact_head(embed1, embed2, mask1, mask2)
        
    def encode_pair(
        self,
        seq1_input_ids: torch.Tensor,
        seq2_input_ids: torch.Tensor,
        seq1_attention_mask: Optional[torch.Tensor] = None,
        seq2_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Encodes a pair of protein sequences through PLM in a single batched call.
        Returns PLM-level representations (residue embeddings and pooled embeddings).
        """
        B = seq1_input_ids.shape[0]
        
        # Batch both sequences together for a single PLM call
        # Concatenate along batch dimension: (2*B, L)
        batched_input_ids = torch.cat([seq1_input_ids, seq2_input_ids], dim=0)
        if seq1_attention_mask is not None and seq2_attention_mask is not None:
            batched_attention_mask = torch.cat([seq1_attention_mask, seq2_attention_mask], dim=0)
        else:
            batched_attention_mask = None
        
        # Single PLM call: (2*B, L, D_plm)
        batched_residue_embeds= self.encode_protein(batched_input_ids, batched_attention_mask)
        
        # Split back into seq1 and seq2
        residue_embed1 = batched_residue_embeds[:B]  # (B, L1, D_plm)
        residue_embed2 = batched_residue_embeds[B:]  # (B, L2, D_plm)
        
        return residue_embed1, residue_embed2
    
    def embed_pair(
        self,
        seq1_input_ids: torch.Tensor,
        seq2_input_ids: torch.Tensor,
        seq1_attention_mask: Optional[torch.Tensor] = None,
        seq2_attention_mask: Optional[torch.Tensor] = None,
        seq1_cluster: Optional[torch.Tensor] = None,
        seq2_cluster: Optional[torch.Tensor] = None,
    ):
        """"
        Returns:
            q1 (torch.Tensor): (B, D_clip) - Q(A) - Normalized query embedding for seq1
            k2 (torch.Tensor): (B, D_clip) - K(B) - Normalized key embedding for seq2
            k1 (torch.Tensor): (B, D_clip) - K(A) - Normalized key embedding for seq1 (for identity check)
            q2 (torch.Tensor): (B, D_clip) - Q(B) - Normalized query embedding for seq2 (for identity check)
            seq1_cluster_out: Processed cluster IDs for seq1 (doubled if swap_self_negative)
            seq2_cluster_out: Processed cluster IDs for seq2 (doubled if swap_self_negative)
        """
        residue_embed1, residue_embed2 = self.encode_pair(
            seq1_input_ids, seq2_input_ids, seq1_attention_mask, seq2_attention_mask
        )
        
        q = self.head_q(residue_embed1, seq1_attention_mask)
        k = self.head_k(residue_embed2, seq2_attention_mask)

        if self.swap_self_negative and self.training:
            q2 = self.head_q(residue_embed2, seq2_attention_mask)
            k1 = self.head_k(residue_embed1, seq1_attention_mask)
            q = torch.cat([q, q2], dim=0)
            k = torch.cat([k, k1], dim=0)
            if seq1_cluster is not None and seq2_cluster is not None:
                seq1_cluster = torch.cat([seq1_cluster, seq2_cluster], dim=0)
                seq2_cluster = torch.cat([seq2_cluster, seq1_cluster[:seq1_cluster.shape[0]//2]], dim=0)

        return q, k, seq1_cluster, seq2_cluster

    def forward(
        self,
        seq1_input_ids: torch.Tensor,
        seq2_input_ids: torch.Tensor,
        seq1_attention_mask: Optional[torch.Tensor] = None,
        seq2_attention_mask: Optional[torch.Tensor] = None,
        seq1_cluster: Optional[torch.Tensor] = None,
        seq2_cluster: Optional[torch.Tensor] = None,
    ):
        """
        Full forward pass for both CLIP and contact prediction.
        """
        seq1_mask = seq1_attention_mask.bool()
        seq2_mask = seq2_attention_mask.bool()
        residue_embed1, residue_embed2 = self.encode_pair(
            seq1_input_ids, seq2_input_ids, seq1_mask, seq2_mask
        )

        q = self.head_q(residue_embed1, seq1_mask)
        k = self.head_k(residue_embed2, seq2_mask)
        if self.swap_self_negative and self.training:
            q2 = self.head_q(residue_embed2, seq2_mask)
            k1 = self.head_k(residue_embed1, seq1_mask)
            q = torch.cat([q, q2], dim=0)
            k = torch.cat([k, k1], dim=0)
            if seq1_cluster is not None and seq2_cluster is not None:
                orig_seq1 = seq1_cluster
                orig_seq2 = seq2_cluster
                seq1_cluster = torch.cat([orig_seq1, orig_seq2], dim=0)
                seq2_cluster = torch.cat([orig_seq2, orig_seq1], dim=0)

        # --- 2. Contact Prediction Path (Residue-level) ---

        if self._adds_special_tokens:
            residue_embed1 = residue_embed1[:, 1:-1, :]
            residue_embed2 = residue_embed2[:, 1:-1, :]
            seq1_mask = seq1_mask[:, 1:-1] if seq1_mask is not None else None
            seq2_mask = seq2_mask[:, 1:-1] if seq2_mask is not None else None

        if self.training:
            neg_embed1, neg_embed2 = residue_embed1, residue_embed2
            neg_mask1, neg_mask2 = seq1_mask, seq2_mask
            neg_clusters1, neg_clusters2 = seq1_cluster, seq2_cluster
            if self.swap_self_negative:
                neg_embed1 = torch.cat([residue_embed1, residue_embed2], dim=0)
                neg_embed2 = torch.cat([residue_embed2, residue_embed1], dim=0)
                neg_mask1 = torch.cat([seq1_mask, seq2_mask], dim=0)
                neg_mask2 = torch.cat([seq2_mask, seq1_mask], dim=0)
            neg1, neg2, neg_mask1, neg_mask2 = get_contact_negatives(
                neg_embed1, neg_embed2, neg_mask1, neg_mask2,
                random_neg_ratio=self.contact_random_neg_ratio,
                self_neg_ratio=self.contact_self_neg_ratio,
                hard_neg_ratio=self.contact_hard_neg_ratio,
                clusters1=neg_clusters1,
                clusters2=neg_clusters2,
                valid_keys=self.valid_keys,
                multiplier=self.multiplier,
                clip_embed1=q,
                clip_embed2=k,
            )
            residue_embed1 = torch.cat([residue_embed1, neg1], dim=0)
            residue_embed2 = torch.cat([residue_embed2, neg2], dim=0)
            seq1_mask = torch.cat([seq1_mask, neg_mask1], dim=0)
            seq2_mask = torch.cat([seq2_mask, neg_mask2], dim=0)
            
        contact_logits, contact_valid_mask = self.predict_contacts(
            residue_embed1, residue_embed2, seq1_mask, seq2_mask
        )

        return q, k, contact_logits, self.logit_scale.exp(), contact_valid_mask, seq1_cluster, seq2_cluster

