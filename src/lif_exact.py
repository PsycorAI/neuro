"""Exact sequential LIF as a Triton kernel (drop-in for lif_parallel).

2026-09-27 finding: lif_parallel's fixed-point iteration never converges at T=512
(every call hits the K=10 cap; 21-31% of spikes differ from the sequential LIF the
docstring describes, and K=3/5/10 disagree with each other on ~41% of spikes). So
the trained neuron is a "10-iteration FPT" function, not a LIF.

This kernel computes the TRUE sequential LIF (same equations as lif_parallel's
docstring: pre[t] = beta*post[t-1] + cur[t]; spk = pre > vth; post = pre - spk*vth,
membrane starts at 0 each call) in ONE memory-bound pass: each program owns a block
of neurons for one batch row and walks t = 0..T-1. Backward uses the same surrogate
as lif_parallel (reset path detached, atan surrogate): grad_pre[t] =
grad_spk[t]/(1+(pi*alpha*(pre-vth))^2) (+ grad_mem_final at t=T-1), then
grad_cur = reverse linear scan with decay beta.

Config: lif_exact: true  (default false = lif_parallel, byte-identical).
lif_reset (gamma, default 1.0 = LIF): post = pre - gamma*spk*vth. gamma=0 is the
no-reset "parallel spiking neuron" (Fang et al. NeurIPS 2023); 0<gamma<1 = partial reset.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _lif_fwd_kernel(cur_ptr, spk_ptr, pre_ptr, memf_ptr, T, N, beta, vth, gamma,
                    BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b * T * N
    mem = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(0, T):
        p = base + t * N + offs
        c = tl.load(cur_ptr + p, mask=mask, other=0.0).to(tl.float32)
        pre = beta * mem + c
        s = (pre > vth).to(tl.float32)
        tl.store(pre_ptr + p, pre, mask=mask)
        tl.store(spk_ptr + p, s.to(spk_ptr.dtype.element_ty), mask=mask)
        mem = pre - gamma * s * vth
    tl.store(memf_ptr + b * N + offs, mem, mask=mask)


@triton.jit
def _lif_bwd_kernel(gspk_ptr, pre_ptr, gmemf_ptr, gcur_ptr, T, N, beta, vth, pa,
                    BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = b * T * N
    acc = tl.load(gmemf_ptr + b * N + offs, mask=mask, other=0.0).to(tl.float32)
    for i in range(0, T):
        t = T - 1 - i
        p = base + t * N + offs
        gs = tl.load(gspk_ptr + p, mask=mask, other=0.0).to(tl.float32)
        o = (tl.load(pre_ptr + p, mask=mask, other=0.0) - vth) * pa
        acc = acc + gs / (1.0 + o * o)          # grad_pre[t] (+ mem_final term folded into acc init)
        tl.store(gcur_ptr + p, acc.to(gcur_ptr.dtype.element_ty), mask=mask)
        acc = acc * beta


class _LIFExact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cur, beta, vth, alpha, gamma):
        cur = cur.contiguous()
        B, T, N = cur.shape
        spk = torch.empty_like(cur)
        pre = torch.empty(B, T, N, device=cur.device, dtype=torch.float32)
        memf = torch.empty(B, N, device=cur.device, dtype=torch.float32)
        BLOCK = 256
        grid = (B, triton.cdiv(N, BLOCK))
        _lif_fwd_kernel[grid](cur, spk, pre, memf, T, N, float(beta), float(vth), float(gamma),
                              BLOCK=BLOCK)
        ctx.save_for_backward(pre)
        ctx.beta, ctx.vth, ctx.alpha, ctx.dtype = float(beta), float(vth), float(alpha), cur.dtype
        return spk, memf.to(cur.dtype)

    @staticmethod
    def backward(ctx, gspk, gmemf):
        (pre,) = ctx.saved_tensors
        B, T, N = pre.shape
        gspk = torch.zeros_like(pre) if gspk is None else gspk.contiguous()
        gmemf = torch.zeros(B, N, device=pre.device) if gmemf is None else gmemf.contiguous()
        # mem_final = post[T-1] = pre[T-1] - spk_hard*vth: reset detached -> d/dpre = 1,
        # so it simply seeds the reverse-scan accumulator at t = T-1.
        gcur = torch.empty(B, T, N, device=pre.device, dtype=ctx.dtype)
        BLOCK = 256
        grid = (B, triton.cdiv(N, BLOCK))
        _lif_bwd_kernel[grid](gspk, pre, gmemf, gcur, T, N, ctx.beta, ctx.vth,
                              math.pi * ctx.alpha, BLOCK=BLOCK)
        return gcur, None, None, None, None


@torch.compiler.disable
def lif_exact(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0):
    """Same signature/returns as lif_parallel (K ignored). spk carries the atan
    surrogate gradient; values are exact {0,1} spikes of the sequential neuron
    (gamma = reset strength; mem_final is post-reset with the same gamma)."""
    return _LIFExact.apply(cur, beta, vth, alpha, gamma)


# --- ppl-gap plan C1/C2/C3 (2026-09-28, docs/ppl_gap_plan_2026_09_27.md) -----------------------------
# Generalized exact LIF, used ONLY when lif_levels > 1, lif_signed or lif_learn_beta is set (the plain
# lif_exact path above is untouched). Per-neuron beta (vector, optionally learnable); integer multi-level
# spikes s = clamp(floor(pre/vth), 0, L) (I-LIF style "integer training, spike inference": s spikes in L
# virtual sub-steps); signed = ternary-style s in [-L, L] (fires -1 when pre <= -vth). Reset subtracts
# gamma*s*vth. Surrogate = staircase of atan bumps at every level (+-j*vth), reset detached as above.
# grad_beta[n] = sum_{b,t} grad_pre[t] * post[t-1].

@triton.jit
def _lif_ml_count(pre, vth, LEVELS: tl.constexpr, SIGNED: tl.constexpr):
    pos = tl.minimum(tl.maximum(tl.floor(pre / vth), 0.0), LEVELS * 1.0)
    if SIGNED:
        neg = tl.minimum(tl.maximum(tl.floor(-pre / vth), 0.0), LEVELS * 1.0)
        pos = pos - neg
    return pos


@triton.jit
def _lif_ml_fwd_kernel(cur_ptr, beta_ptr, spk_ptr, pre_ptr, memf_ptr, T, N, vth, gamma,
                       LEVELS: tl.constexpr, SIGNED: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    base = b * T * N
    mem = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(0, T):
        p = base + t * N + offs
        c = tl.load(cur_ptr + p, mask=mask, other=0.0).to(tl.float32)
        pre = beta * mem + c
        s = _lif_ml_count(pre, vth, LEVELS, SIGNED)
        tl.store(pre_ptr + p, pre, mask=mask)
        tl.store(spk_ptr + p, s.to(spk_ptr.dtype.element_ty), mask=mask)
        mem = pre - gamma * s * vth
    tl.store(memf_ptr + b * N + offs, mem, mask=mask)


@triton.jit
def _lif_ml_bwd_kernel(gspk_ptr, pre_ptr, beta_ptr, gmemf_ptr, gcur_ptr, gbeta_ptr, T, N, vth, pa, gamma,
                       LEVELS: tl.constexpr, SIGNED: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    base = b * T * N
    acc = tl.load(gmemf_ptr + b * N + offs, mask=mask, other=0.0).to(tl.float32)
    gb = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, T):
        t = T - 1 - i
        p = base + t * N + offs
        gs = tl.load(gspk_ptr + p, mask=mask, other=0.0).to(tl.float32)
        pre = tl.load(pre_ptr + p, mask=mask, other=0.0)
        bump = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(1, LEVELS + 1):
            o = (pre - j * vth) * pa
            bump += 1.0 / (1.0 + o * o)
            if SIGNED:
                o2 = (pre + j * vth) * pa
                bump += 1.0 / (1.0 + o2 * o2)
        acc = acc + gs * bump                                   # grad wrt pre[t]
        tl.store(gcur_ptr + p, acc.to(gcur_ptr.dtype.element_ty), mask=mask)
        pp = tl.load(pre_ptr + p - N, mask=mask & (t > 0), other=0.0)   # pre[t-1] (0 at t=0)
        post_prev = pp - gamma * _lif_ml_count(pp, vth, LEVELS, SIGNED) * vth
        gb += acc * post_prev
        acc = acc * beta
    tl.store(gbeta_ptr + b * N + offs, gb, mask=mask)


class _LIFExactML(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cur, beta_vec, vth, alpha, gamma, levels, signed):
        cur = cur.contiguous()
        B, T, N = cur.shape
        beta_c = beta_vec.detach().float().contiguous()
        spk = torch.empty_like(cur)
        pre = torch.empty(B, T, N, device=cur.device, dtype=torch.float32)
        memf = torch.empty(B, N, device=cur.device, dtype=torch.float32)
        BLOCK = 256
        grid = (B, triton.cdiv(N, BLOCK))
        _lif_ml_fwd_kernel[grid](cur, beta_c, spk, pre, memf, T, N, float(vth), float(gamma),
                                 LEVELS=int(levels), SIGNED=bool(signed), BLOCK=BLOCK)
        ctx.save_for_backward(pre, beta_c)
        ctx.vth, ctx.alpha, ctx.gamma, ctx.levels, ctx.signed = float(vth), float(alpha), float(gamma), int(levels), bool(signed)
        ctx.dtype, ctx.beta_dtype = cur.dtype, beta_vec.dtype
        return spk, memf.to(cur.dtype)

    @staticmethod
    def backward(ctx, gspk, gmemf):
        pre, beta_c = ctx.saved_tensors
        B, T, N = pre.shape
        gspk = torch.zeros_like(pre) if gspk is None else gspk.contiguous()
        gmemf = torch.zeros(B, N, device=pre.device) if gmemf is None else gmemf.contiguous()
        gcur = torch.empty(B, T, N, device=pre.device, dtype=ctx.dtype)
        gbeta = torch.empty(B, N, device=pre.device, dtype=torch.float32)
        BLOCK = 256
        grid = (B, triton.cdiv(N, BLOCK))
        _lif_ml_bwd_kernel[grid](gspk, pre, beta_c, gmemf, gcur, gbeta, T, N, ctx.vth,
                                 math.pi * ctx.alpha, ctx.gamma,
                                 LEVELS=ctx.levels, SIGNED=ctx.signed, BLOCK=BLOCK)
        gb = gbeta.sum(0).to(ctx.beta_dtype) if ctx.needs_input_grad[1] else None
        return gcur, gb, None, None, None, None, None


@torch.compiler.disable
def lif_exact_ml(cur, beta, vth=1.0, K=None, alpha=2.0, gamma=1.0, levels=1, signed=False):
    """lif_exact generalization (see block comment above). beta: float or (N,) tensor (learnable if it
    requires grad). Returns (spk (B,T,N) integer-valued with staircase surrogate grad, mem_final)."""
    if not torch.is_tensor(beta):
        beta = torch.full((cur.shape[-1],), float(beta), device=cur.device)
    return _LIFExactML.apply(cur, beta, vth, alpha, gamma, levels, signed)
