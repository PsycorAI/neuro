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
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _lif_fwd_kernel(cur_ptr, spk_ptr, pre_ptr, memf_ptr, T, N, beta, vth,
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
        mem = pre - s * vth
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
    def forward(ctx, cur, beta, vth, alpha):
        cur = cur.contiguous()
        B, T, N = cur.shape
        spk = torch.empty_like(cur)
        pre = torch.empty(B, T, N, device=cur.device, dtype=torch.float32)
        memf = torch.empty(B, N, device=cur.device, dtype=torch.float32)
        BLOCK = 256
        grid = (B, triton.cdiv(N, BLOCK))
        _lif_fwd_kernel[grid](cur, spk, pre, memf, T, N, float(beta), float(vth), BLOCK=BLOCK)
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
        return gcur, None, None, None


@torch.compiler.disable
def lif_exact(cur, beta, vth=1.0, K=None, alpha=2.0):
    """Same signature/returns as lif_parallel (K ignored). spk carries the atan
    surrogate gradient; values are exact {0,1} spikes of the sequential LIF."""
    return _LIFExact.apply(cur, beta, vth, alpha)
