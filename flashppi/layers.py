import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint
from einops import repeat


def swiglu(x, y):
    return F.silu(x) * y


def rmsnorm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer("variance_epsilon", torch.tensor(eps), persistent=False)

    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)


class MLPHead(nn.Module):
    """SwiGLU MLP head."""

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: float = 2.0):
        super().__init__()
        hidden_dim = int(in_dim * hidden_mult)
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerConfig:
    def __init__(self, dim, heads, depth, out_dim, ffn_mult=None, use_rope=True):
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.swiglu_multiple_of = 256
        self.ffn_dim_multiplier = ffn_mult
        self.out_dim = out_dim
        self.norm_eps = 1e-6
        self.use_rope = use_rope


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin, position_ids=None, interleaved=False):
    """Apply rotary embeddings to x (B, L, H, head_dim)."""
    if position_ids is not None:
        cos = cos[position_ids]   # (B, L, D/2)
        sin = sin[position_ids]
    else:
        cos = cos[:x.shape[1]]    # (L, D/2)
        sin = sin[:x.shape[1]]

    if not interleaved:
        cos = repeat(cos, "... d -> ... 1 (2 d)")
        sin = repeat(sin, "... d -> ... 1 (2 d)")
    else:
        cos = repeat(cos, "... d -> ... 1 (d 2)")
        sin = repeat(sin, "... d -> ... 1 (d 2)")

    ro_dim = cos.shape[-1]
    return torch.cat([
        x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin,
        x[..., ro_dim:],
    ], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if seqlen > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        # q, k: (B, L, H, head_dim)
        seqlen = position_ids.max().item() + 1 if position_ids is not None else q.shape[1]
        self._update_cos_sin_cache(seqlen, device=q.device, dtype=q.dtype)
        q = apply_rotary_emb(q, self._cos_cached, self._sin_cached, position_ids)
        k = apply_rotary_emb(k, self._cos_cached, self._sin_cached, position_ids)
        return q, k


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.heads
        self.head_dim = config.dim // config.heads
        self.wqkv = nn.Linear(config.dim, self.n_heads * self.head_dim * 3, bias=False)
        self.wo = nn.Linear(config.heads * self.head_dim, config.dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim) if config.use_rope else None

    def forward(self, x, attention_mask=None, position_ids=None):
        bsz, seqlen, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = torch.split(qkv, self.n_heads * self.head_dim, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool() if attention_mask is not None else None
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_heads * self.head_dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config.dim, 4 * config.dim, config.swiglu_multiple_of, config.ffn_dim_multiplier)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x, attention_mask=None, position_ids=None):
        h = x + self.attention(self.attention_norm(x), attention_mask, position_ids)
        return h + self.feed_forward(self.ffn_norm(h))


class TransformerLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.depth)])
        self.apply(self._init_weights)
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, attention_mask=None):
        position_ids = None
        if attention_mask is not None:
            mask_long = attention_mask.long()
            position_ids = (mask_long.cumsum(dim=1) - 1).clamp(min=0)

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(layer, x, attention_mask, position_ids, use_reentrant=False)
            else:
                x = layer(x, attention_mask, position_ids)

        return x
