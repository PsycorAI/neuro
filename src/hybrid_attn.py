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


class SpikeGLU(nn.Module):
    """r15b (2026-10-01, docs/interpretable_spiking_plan_2026_10_01.md S1): spike-gated GLU FFN.
    a = W1 x. Each token keeps only its k largest |a| units (k-winner-take-all = lateral inhibition),
    and each surviving unit emits a graded spike a * (W3 x) (Loihi-2 style graded payload):
    h = kWTA(a) * (W3 x), out = W2 h. Exactly <= k active hidden units per token, so the FFN's
    features can be inspected one unit at a time. Same 3-matrix layout as SwiGLU (hidden ~ 8d/3
    param-matches the 4d GELU MLP). Gradient flows through the selected units only (TopK-SAE style)."""
    def __init__(self, d, hidden=None, k=64, bias=False, lif=False, lif_levels=1, lif_signed=True,
                 lif_beta_init=0.1):
        super().__init__()
        if hidden is None:
            hidden = ((int(8 * d / 3) + 63) // 64) * 64
        self.k = min(int(k), hidden)
        self.w1 = nn.Linear(d, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, d, bias=bias)
        self.w3 = nn.Linear(d, hidden, bias=bias)
        # r16 b3 (2026-10-02, docs/v4_plan_review_2026_10_01.md F2): k_cur = per-step active-unit budget set by
        # the trainer (spikeglu_k_at: dense until a start fraction, then annealed to k). None = self.k.
        self.k_cur = None
        # r16 b4 "DSN-LIF" (default off): the apical input a = W1 x drives a LIF neuron (membrane with a
        # learnable per-neuron leak, threshold 1, reset; signed lif_levels spikes) instead of k-WTA. The
        # neuron's spike s gates the graded basal payload: h = s * (W3 x). Firing is set by the threshold and
        # the trainer's sparsity penalty (the spike rate is reported via _last_rate).
        self.lif = bool(lif)
        self._last_rate = None
        if self.lif:
            self.lif_levels, self.lif_signed = int(lif_levels), bool(lif_signed)
            self.beta_logit = nn.Parameter(torch.full((hidden,), math.log(lif_beta_init / (1 - lif_beta_init))))
    def forward(self, x):
        a = self.w1(x)
        if self.lif:
            import lif_exact
            s, _ = lif_exact.lif_exact_ml(a, torch.sigmoid(self.beta_logit), vth=1.0,
                                          levels=self.lif_levels, signed=self.lif_signed)
            self._last_rate = s.abs().mean()
            return self.w2(s * self.w3(x))
        a = _kwta(self, a)
        return self.w2(a * self.w3(x))


class SparseAct(nn.Module):
    """r19 (2026-10-03, sparse "best of both worlds" MLP; default off). Activation for the plain 2-matrix MLP:
    kind "relu2" = relu(a)^2 (Primer); "relu" = relu(a). Optional homeostatic threshold (homeo = "layer": one
    threshold per block, "unit": one per hidden neuron): a neuron fires only when a > theta, and emits a graded
    value (relu2/relu of a). theta is NOT trained by gradient: after each optimizer step the trainer calls
    homeo_step(), which nudges theta so the firing rate tracks rate_target (set per step by the trainer; 1.0 =
    no threshold pressure). Gradient flows through the units that fired. Rates are fractions of nonzero outputs."""
    def __init__(self, hidden, kind="relu2", homeo=None):
        super().__init__()
        assert kind in ("relu2", "relu") and homeo in (None, "layer", "unit")
        self.kind, self.homeo = kind, homeo
        self.rate_target = 1.0
        # Rate bookkeeping lives in non-persistent buffers updated IN PLACE (2026-10-03): no Python state changes
        # inside forward, so torch.compile does not guard/recompile on it and CUDA-graph capture is safe. Same math
        # as the first (Python-counter) version, bit-identical (temp/parity_sparseact.py).
        self.register_buffer("last_rate", torch.zeros(()), persistent=False)
        if homeo:
            self.register_buffer("theta", torch.zeros(hidden if homeo == "unit" else 1))
            self.register_buffer("rate_acc", torch.zeros(hidden if homeo == "unit" else 1), persistent=False)
            self.register_buffer("rate_n", torch.zeros(()), persistent=False)
    @property
    def _last_rate(self):                     # name used by the trainer telemetry and the scorecard
        return self.last_rate
    def forward(self, a):
        fire = a > (self.theta.to(a.dtype) if self.homeo else 0)
        r = torch.where(fire, a, torch.zeros_like(a))
        h = r * r if self.kind == "relu2" else r
        with torch.no_grad():
            f = fire.float()
            self.last_rate.copy_(f.mean())
            if self.homeo and self.training:
                u = f.reshape(-1, f.shape[-1]).mean(0) if self.homeo == "unit" else f.mean().reshape(1)
                self.rate_acc.add_(u)
                self.rate_n.add_(1)
        return h
    @torch.no_grad()
    def homeo_step(self, lr):
        """theta += lr * (rate - target) / target, clamped at 0 (relative error, so the step size is target-free).
        No host sync: a step with no training forward (rate_n == 0) adds nothing."""
        if not self.homeo:
            return
        upd = lr * (self.rate_acc / self.rate_n.clamp(min=1) - self.rate_target) / self.rate_target
        self.theta.add_(torch.where(self.rate_n > 0, upd, torch.zeros_like(upd))).clamp_(min=0)
        self.rate_acc.zero_(); self.rate_n.zero_()


@torch.compiler.disable
def _kwta(ffn, a):
    """k-WTA on the apical input. Compiler-disabled and reads ffn.k_cur itself (2026-10-02 speed plan): the k-ramp
    sets a new Python int almost every step, which would recompile any compiled graph that reads it. Eager = same math."""
    k = ffn.k if ffn.k_cur is None else min(int(ffn.k_cur), a.shape[-1])
    if k < a.shape[-1]:
        thr = a.detach().abs().topk(k, dim=-1).values[..., -1:]
        a = a * (a.detach().abs() >= thr)
    return a


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
                 use_qk_norm=True, use_ffn=True, spikeglu_k=0, topk=0,
                 out_gate=False, canon=0, spikeglu_lif=False, lif_levels=1, lif_signed=True,
                 ffn_act="gelu", ffn_homeo=None):
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
        # r15 flags (2026-10-01, default off = byte-identical): use_ffn=False drops this block's MLP
        # (attention only routes, r15a); spikeglu_k>0 swaps the MLP for a k-WTA SpikeGLU (r15b);
        # topk>0 = hard top-k attention, softmax over each query's k best causal keys (r15c).
        self.attn_topk = int(topk)
        if not use_ffn:
            self.norm_ffn = None
            self.ffn = None
            out_linear = None
        else:
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
        if not use_ffn:
            pass
        elif spikeglu_k:
            self.ffn = SpikeGLU(d_mem, hidden=ffn_hidden, k=spikeglu_k, bias=bias,
                                lif=spikeglu_lif, lif_levels=lif_levels, lif_signed=lif_signed)
            out_linear = self.ffn.w2
        elif use_swiglu:
            self.ffn = SwiGLU(d_mem, hidden=ffn_hidden, bias=bias)
            out_linear = self.ffn.w2
        else:
            hidden = ffn_hidden or (4 * d_mem)
            # r19 ffn_act (default "gelu" = byte-identical): "relu2"/"relu" = SparseAct (no params, same state_dict
            # names apart from the homeostatic theta buffer).
            self.ffn = nn.Sequential(nn.Linear(d_mem, hidden, bias=bias),
                                     nn.GELU() if ffn_act == "gelu" else SparseAct(hidden, ffn_act, ffn_homeo),
                                     nn.Linear(hidden, d_mem, bias=bias))
            out_linear = self.ffn[-1]
        # Depth-scaled init on the two residual-adjacent output projections
        # (matches SpikingHebbianBlock convention).
        if depth_scale != 1.0:
            with torch.no_grad():
                self.o_proj.weight.mul_(depth_scale)
                if out_linear is not None:
                    out_linear.weight.mul_(depth_scale)

        # RoPE tables (register as buffers so they move with .to(device))
        cos, sin = _rope_freqs(self.d_head, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # attn_window (2026-09-29, docs/contiguous_training_design.md): 0 = full causal (default, byte-identical).
        # W > 0 = sliding window, each query sees keys t-W+1..t. Set per step by the trainer (stochastic
        # SWAX-style windowing); not a constructor arg so checkpoints and build_model are unchanged.
        self.attn_window = 0
        self._band_cache = {}
        # r16 flags (2026-10-02, docs/v4_plan_review_2026_10_01.md 6.5; default off = byte-identical; created
        # last so existing parameter inits are unchanged even when on).
        # b2 out_gate: elementwise sigmoid gate on the attention output before o_proj, computed from the
        #    normed block input (Qwen gated attention G1, arXiv 2505.06708: removes attention sinks, which
        #    would otherwise take a top-k slot in every query).
        # b7 canon=k: Canon-A causal depthwise conv (zero-init taps) on the attention input; the residual
        #    still carries the raw x (as SpikingHebbianBlock canon_a).
        self.attn_gate = nn.Linear(d_mem, d_mem, bias=bias) if out_gate else None
        self.attn_canon = int(canon)
        if self.attn_canon:
            self.canon_w = nn.Parameter(torch.zeros(d_mem, 1, self.attn_canon))

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
        w = self.attn_window
        if self.attn_topk:
            att = self._topk_weights(q, k, w=w if 0 < w < T else 0)
            out = torch.matmul(att.to(v.dtype), v)
        elif 0 < w < T:
            key = (T, w, q.device)
            band = self._band_cache.get(key)
            if band is None:
                i = torch.arange(T, device=q.device)
                diff = i[:, None] - i[None, :]                    # query - key
                band = self._band_cache[key] = (diff >= 0) & (diff < w)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=band)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        if self.attn_gate is not None:
            out = out * torch.sigmoid(self.attn_gate(x))
        return self.o_proj(out)

    def _topk_weights(self, q, k, qpos=None, kpos=None, w=0):
        """r15c: hard top-k attention weights (B, H, tq, tk). Scores in fp32, causal (and optional
        window w) mask, keep each query's attn_topk best keys, softmax over the survivors. A query
        with fewer than k visible keys keeps them all. Explicit T x T scores (no flash kernel)."""
        s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        if qpos is None:
            qpos = torch.arange(q.shape[-2], device=q.device)
            kpos = torch.arange(k.shape[-2], device=q.device)
        diff = qpos[:, None] - kpos[None, :]
        allowed = diff >= 0
        if w > 0:
            allowed = allowed & (diff < w)
        s = s.masked_fill(~allowed, float("-inf"))
        kth = s.topk(min(self.attn_topk, s.shape[-1]), dim=-1).values[..., -1:]
        return torch.softmax(s.masked_fill(s < kth, float("-inf")), dim=-1)

    def _attn_weights(self, x):
        """Audit/diagnostics (not used in training): per-head attention weights (B, H, T, T) for a
        normed input x -- top-k weights when attn_topk is set, else dense causal softmax."""
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q, k, _ = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        if self.attn_topk:
            return self._topk_weights(q, k)
        s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * Dh ** -0.5
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        return torch.softmax(s, dim=-1)

    def forward_sequence(self, x_seq, mem, M, prev_spk,
                          ablate_memory=False, fpt_K=10):
        """Same signature as SpikingHebbianBlock.forward_sequence.

        Returns (out_seq, mem, M, spk_seq, final_spk). The state slots pass
        through unchanged; we manufacture a zero spike sequence at the right
        shape so downstream spike-rate telemetry doesn't crash.
        """
        # attention + FFN with pre-norm residuals
        x_a = x_seq
        if self.attn_canon:
            x_a = x_seq + F.conv1d(F.pad(x_seq.transpose(1, 2), (self.attn_canon - 1, 0)),
                                   self.canon_w.to(x_seq.dtype), groups=x_seq.size(-1)).transpose(1, 2)
        y = x_seq + self._attn(self.norm_attn(x_a))
        z = y if self.ffn is None else y + self.ffn(self.norm_ffn(y))
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
        if self.attn_topk:
            out = torch.matmul(self._topk_weights(q, k, qpos=qpos, kpos=kpos).to(v.dtype), v)
        else:
            keep = kpos[None, :] <= qpos[:, None]                   # (t, k_len) bool
            attn_mask = torch.zeros(t, k_len, dtype=q.dtype, device=x.device)
            attn_mask.masked_fill_(~keep, float("-inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, t, D)
        if self.attn_gate is not None:
            out = out * torch.sigmoid(self.attn_gate(x))
        return self.o_proj(out), new_cache

    def step_kv(self, x, cache=None):
        """Pre-norm residual block over new tokens `x` (B, t, D) with a KV cache.
        Returns (z (B, t, D), new_cache). Drop-in for the attention layer inside
        an incremental-decode loop; state slots (mem/M/prev_spk) are irrelevant
        to attention and handled by the spiking blocks' own initial_state path.
        """
        if self.attn_canon:
            raise NotImplementedError("step_kv with attn_canon needs the last k-1 inputs as state "
                                      "(streaming TO-BUILD, docs/v4_plan_review_2026_10_01.md 6.6)")
        a, new_cache = self._attn_kv(self.norm_attn(x), cache)
        y = x + a
        z = y if self.ffn is None else y + self.ffn(self.norm_ffn(y))
        return z, new_cache
