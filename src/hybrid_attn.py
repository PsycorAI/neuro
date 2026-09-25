"""Hybrid attention block — a standard Transformer block (softmax MHA + RoPE
+ SwiGLU FFN) that can be interleaved with SpikingHebbianBlocks in the same
stack. Follows the SpikingBrain-2025 / MiniMax-01 / Jamba pattern of putting
full attention every ~N layers among linear/state-space layers.

The Hebbian memory in SpikingHebbianBlock is a linear-attention-class mechanism
that decays context into a fixed-size matrix. That's cheap and gives the
project its persistent-brain differentiator, but pure linear-attention
underperforms on short-context coherence and exact retrieval. Interleaving one
or two full-attention layers restores those and keeps the Hebbian layers doing
the compression / persistence work they're actually good at.

Interface: forward_sequence(x_seq, mem, M, prev_spk, ablate_memory, fpt_K)
matches SpikingHebbianBlock so the LM's block loop needs no branching. The
state slots (mem, M, prev_spk) are passed through unchanged since attention has
no persistent recurrent state — save_brain/load_brain works transparently.

Not implemented (yet): step() for token-by-token generation, since chat.py
processes the last `block_size` tokens per token generated (see
scripts/chat.py generate_response). If we later need true KV-cache
streaming, add step() with a growing KV cache. For now the block runs the
whole sequence each step, same as SpikingHebbianBlock's forward_sequence path.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        inv = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * inv).to(x.dtype) * self.weight


def _rope_freqs(head_dim, max_seq_len, base=10000.0, device=None, dtype=torch.float32):
    """Precompute cos/sin tables for RoPE. Returns (cos, sin) shape (T, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)                           # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                    # (T, head_dim)
    return emb.cos(), emb.sin()


def _apply_rope(x, cos, sin):
    """Apply RoPE to (B, H, T, D). cos/sin are (T, D)."""
    # rotate_half: split last dim, negate second half, swap
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    T = x.shape[-2]
    c = cos[:T].to(x.dtype).view(1, 1, T, -1)
    s = sin[:T].to(x.dtype).view(1, 1, T, -1)
    return x * c + rotated * s


class SwiGLU(nn.Module):
    """Standard SwiGLU FFN. hidden = 2/3 * 4d rounded to multiple of 64 by default."""
    def __init__(self, d, hidden=None, bias=False):
        super().__init__()
        if hidden is None:
            hidden = ((int(8 * d / 3) + 63) // 64) * 64
        self.w1 = nn.Linear(d, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, d, bias=bias)
        self.w3 = nn.Linear(d, hidden, bias=bias)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HybridAttentionBlock(nn.Module):
    """Drop-in Transformer block matching SpikingHebbianBlock's interface.

    Architecture:
        y = x + attn(rmsnorm(x))
        z = y + swiglu(rmsnorm(y))

    Inputs and outputs are (B, T, d_in==d_mem). State slots are pass-through.
    """
    def __init__(self, d_in, d_mem, n_neurons, n_heads=8, use_rmsnorm=True,
                 use_swiglu=True, depth_scale=1.0, max_seq_len=4096,
                 rope_base=10000.0, ffn_hidden=None, bias=False,
                 use_qk_norm=True):
        super().__init__()
        if d_in != d_mem:
            # Attention block currently only supports the mid-stack case where
            # residual-stream dim equals d_mem (n_layers > 1). This matches
            # every non-first Spiking block; the first block adapts d -> d_mem
            # and hybrid attention wouldn't be placed there anyway.
            raise ValueError(f"HybridAttentionBlock requires d_in == d_mem "
                             f"(got d_in={d_in}, d_mem={d_mem}); place attention "
                             f"only at layer_idx > 0.")
        if d_mem % n_heads != 0:
            raise ValueError(f"d_mem ({d_mem}) must be divisible by n_heads ({n_heads})")
        self.d = d_mem
        self.n_neurons = n_neurons              # for building compatible spk stub
        self.n_heads = n_heads
        self.d_head = d_mem // n_heads
        self.max_seq_len = max_seq_len

        # Dummy state metadata so the LM's state allocator produces (B,1,1)
        # tensors we can ignore — cleanest way to keep block-loop uniform.
        self.n_state_units = 1
        self.d_state = 1

        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm_attn = norm_cls(d_mem)
        self.norm_ffn = norm_cls(d_mem)
        # QK-norm: per-head RMSNorm on Q and K (over d_head) before the dot
        # product. Bounds attention-logit magnitude and — per the 2026-08-01
        # seed-variance finding, which localized all seed-to-seed variance to
        # the attention layer — is the intended fix to make the hybrid recipe
        # seed-robust. Always RMSNorm (QK-norm convention), independent of the
        # residual-stream norm choice above.
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)
        self.qkv = nn.Linear(d_mem, 3 * d_mem, bias=bias)
        self.o_proj = nn.Linear(d_mem, d_mem, bias=bias)
        if use_swiglu:
            self.ffn = SwiGLU(d_mem, hidden=ffn_hidden, bias=bias)
            out_linear = self.ffn.w2
        else:
            hidden = ffn_hidden or (4 * d_mem)
            self.ffn = nn.Sequential(nn.Linear(d_mem, hidden, bias=bias),
                                     nn.GELU(),
                                     nn.Linear(hidden, d_mem, bias=bias))
            out_linear = self.ffn[-1]
        # Depth-scaled init on the two residual-adjacent output projections
        # (matches SpikingHebbianBlock convention).
        if depth_scale != 1.0:
            with torch.no_grad():
                self.o_proj.weight.mul_(depth_scale)
                out_linear.weight.mul_(depth_scale)

        # RoPE tables (register as buffers so they move with .to(device))
        cos, sin = _rope_freqs(self.d_head, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _attn(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)                                        # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        H, Dh = self.n_heads, self.d_head
        # (B, T, D) -> (B, H, T, Dh)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        # QK-norm before RoPE (canonical order: normalize the learned Q/K
        # directions, then apply positional rotation).
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        # Causal SDPA (Flash if available). No mask needed with is_causal=True.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

    def forward_sequence(self, x_seq, mem, M, prev_spk,
                          ablate_memory=False, fpt_K=10):
        """Same signature as SpikingHebbianBlock.forward_sequence.

        Returns (out_seq, mem, M, spk_seq, final_spk). The state slots pass
        through unchanged; we manufacture a zero spike sequence at the right
        shape so downstream spike-rate telemetry doesn't crash.
        """
        # attention + FFN with pre-norm residuals
        y = x_seq + self._attn(self.norm_attn(x_seq))
        z = y + self.ffn(self.norm_ffn(y))
        # State pass-through: attention has no persistent recurrent state.
        # `mem`/`prev_spk` are (B, n_neurons) allocated at LM top and shared
        # in shape across all layers; we pass them through unchanged so
        # brain-save/load stays uniform. `M` is (B, 1, 1) from our dummy
        # d_state metadata. spk stub is (B, T, n_neurons) zeros so shape
        # matches spiking layers for downstream stats — the mean spike rate
        # will be slightly under-reported (attention layers count as 0),
        # which effectively LOOSENS the sparsity penalty for the spiking
        # layers by n_spiking/n_total. Documented tradeoff; acceptable given
        # we're relaxing the sparsity target regardless.
        B, T = x_seq.shape[0], x_seq.shape[1]
        spk_zero = torch.zeros(B, T, self.n_neurons,
                                device=x_seq.device, dtype=x_seq.dtype)
        return z, mem, M, spk_zero, prev_spk

    # --- Incremental KV-cache path (inference/generation only) -----------------
    # Additive: nothing above (the training forward_sequence path) calls these.
    # Enables token-by-token decoding without re-forwarding the whole prefix,
    # which is the OPD-rollout / tool_eval / abstain_eval / self_refine / TTFT
    # bottleneck (the Hebbian/GDN-2 blocks already decode incrementally via
    # forward(initial_state=...); attention was the only layer that couldn't).
    def _attn_kv(self, x, cache):
        """Attention over new tokens `x` (B, t, D) that also attend to cached
        past keys/values. `cache` is (k, v) each (B, H, past, Dh) or None.
        Returns (out (B, t, D), new_cache). Supports t>=1 so the same method
        prefills a prompt (t=prompt_len) and then steps one token at a time.
        """
        B, t, D = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, t, H, Dh).transpose(1, 2)                 # (B, H, t, Dh)
        k = k.view(B, t, H, Dh).transpose(1, 2)
        v = v.view(B, t, H, Dh).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # RoPE at ABSOLUTE positions [past, past+t) -- the whole point of the
        # cache is that the new tokens keep their true sequence positions.
        past = 0 if cache is None else cache[0].shape[2]
        q = _apply_rope(q, self.rope_cos[past:past + t], self.rope_sin[past:past + t])
        k = _apply_rope(k, self.rope_cos[past:past + t], self.rope_sin[past:past + t])
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)                 # (B, H, past+t, Dh)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        # Explicit causal mask, NOT is_causal: SDPA's is_causal aligns the
        # triangle TOP-LEFT, so with a cache (q_len<k_len) a single query would
        # see only key 0. We need bottom-right: new query at absolute position
        # (past+i) attends every key j <= past+i. For t==1 that's all keys; for
        # past==0,t==T it reduces to the standard lower-triangular mask.
        k_len = k.shape[2]
        qpos = torch.arange(t, device=x.device) + past
        kpos = torch.arange(k_len, device=x.device)
        keep = kpos[None, :] <= qpos[:, None]                   # (t, k_len) bool
        attn_mask = torch.zeros(t, k_len, dtype=q.dtype, device=x.device)
        attn_mask.masked_fill_(~keep, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, t, D)
        return self.o_proj(out), new_cache

    def step_kv(self, x, cache=None):
        """Pre-norm residual block over new tokens `x` (B, t, D) with a KV cache.
        Returns (z (B, t, D), new_cache). Drop-in for the attention layer inside
        an incremental-decode loop; state slots (mem/M/prev_spk) are irrelevant
        to attention and handled by the spiking blocks' own initial_state path.
        """
        a, new_cache = self._attn_kv(self.norm_attn(x), cache)
        y = x + a
        z = y + self.ffn(self.norm_ffn(y))
        return z, new_cache
