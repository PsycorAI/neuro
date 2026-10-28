"""sparse_gemv.py -- 2026-10-25 (Neuro 78, Demo plan v2 W3, Glenn: "start W3"). Batch-1 decode GEMV that reads ONLY the
weight rows of NONZERO inputs (v4's spike-coded inputs are exact zeros: skipping them is mathematically exact; only the
summation order differs from the dense GEMV). Used for the Linears whose inputs are mostly zero on real demo turns
(temp/n78_meter_breakdown.py: FFN down 97% zero, attention output 80%, memory block 83%).
  * weights stored TRANSPOSED (in x out, contiguous; the Linear's .weight stays a (out x in) view of it, so dense
    passes, brain merges and the meter's hooks see the same matrix) -> one input's weights are one contiguous row;
  * nonzero inputs compacted into a fixed-size index buffer + a count, all on the GPU (no host sync: CUDA-graph safe);
  * split-K Triton kernel: SPLITS slices of the nonzero rows x BLOCK_N output columns per program, fp32 accumulate,
    partial sums written per split and summed in a fixed order (deterministic, no atomics).
  enable(model, pred) -> number of Linears switched; a switched Linear uses the sparse path only for ONE token (x of
  shape (1, 1, in)) and the dense cuBLAS path otherwise."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SPLITS = 16


@triton.jit
def _sparse_gemv_kernel(x_ptr, wt_ptr, part_ptr, IN, OUT,
                        SPLITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """v2 (n78 10-25): no compaction pass. Each program walks its slice of the inputs; the weight rows of ZERO inputs are
    masked off the load, and predicated-off lanes issue no memory transaction, so only nonzero inputs' rows are read
    (v1's index compaction cost ~5 extra kernels / layer and lost to cuBLAS on the small matrices)."""
    pn = tl.program_id(0); ps = tl.program_id(1)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < OUT
    per = (IN + SPLITS - 1) // SPLITS
    k_lo = ps * per
    k_hi = tl.minimum(k_lo + per, IN)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_ptr + kk, mask=kk < k_hi, other=0.0).to(tl.float32)
        live = (kk < k_hi) & (xv != 0.0)
        w = tl.load(wt_ptr + kk[:, None].to(tl.int64) * OUT + n[None, :], mask=live[:, None] & nmask[None, :],
                    other=0.0).to(tl.float32)
        acc += tl.sum(w * xv[:, None], axis=0)
    tl.store(part_ptr + ps * OUT + n, acc, mask=nmask)


@triton.jit
def _reduce_kernel(part_ptr, b_ptr, y_ptr, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Sum the SPLITS partial rows in a fixed order (deterministic) + bias, cast to the output dtype."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < OUT
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in range(SPLITS):
        acc += tl.load(part_ptr + s * OUT + n, mask=m, other=0.0)
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + n, acc.to(y_ptr.dtype.element_ty), mask=m)


# tiling: temp/n78_sparse_sweep.py on the 5080, the 26 target Linears back to back (DRAM-cold): dense cuBLAS 872 us,
# this 158 us (5.5x); logs/n78_sparse_sweep2.log
CFG = dict(rows_per_split=256, block_n=32, block_k=128, warps=1)


def sparse_linear_1tok(x, wt, bias, ar=None):
    """x: (in,) one token's input; wt: (in, out) contiguous. Returns (out,) = x @ wt (+ bias), reading only the
    weight rows of nonzero x. Two launches: the masked split-K GEMV and the fixed-order reduction. The input is cut
    into slices of CFG rows_per_split (more, shorter slices = more loads in flight: each program's loop is latency-
    bound, not bandwidth-bound)."""
    IN, OUT = wt.shape
    S = max(1, triton.cdiv(IN, CFG["rows_per_split"]))
    part = torch.empty(S, OUT, dtype=torch.float32, device=x.device)
    BN = CFG["block_n"]
    _sparse_gemv_kernel[(triton.cdiv(OUT, BN), S)](x, wt, part, IN, OUT, SPLITS=S, BLOCK_N=BN, BLOCK_K=CFG["block_k"],
                                                     num_warps=CFG["warps"])
    y = torch.empty(OUT, dtype=x.dtype, device=x.device)
    _reduce_kernel[(triton.cdiv(OUT, 256),)](part, bias if bias is not None else part, y, OUT, SPLITS=S,
                                             BLOCK=256, HAS_BIAS=bias is not None)
    return y


def _enable_one(lin):
    W = lin.weight.data
    wt = W.t().contiguous()                                             # (in, out) storage
    lin.weight = nn.Parameter(wt.t(), requires_grad=False)              # (out, in) view of it: same matrix
    lin._sp_ar = torch.arange(W.shape[1], dtype=torch.int32, device=W.device)
    lin._sp_on = True

    def forward(x, _lin=lin):
        if _lin._sp_on and x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1 and _lin.in_features > 0:
            import demo_ops                                             # noqa: F401  torch.ops.n78.sparse_linear
            wt_ = _lin.weight.t()                                       # contiguous (in, out)
            return torch.ops.n78.sparse_linear(x[0, 0], wt_, _lin.bias).view(1, 1, -1)   # opaque to torch.compile
        return F.linear(x, _lin.weight, _lin.bias)
    lin.forward = forward


def enable(model, pred):
    """Switch every nn.Linear named n with pred(n) to the sparse 1-token path; returns how many."""
    k = 0
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and pred(n) and mod.in_features > 0 and mod.out_features > 0:
            _enable_one(mod); k += 1
    return k


def set_enabled(model, on):
    for mod in model.modules():
        if hasattr(mod, "_sp_on"):
            mod._sp_on = on


def demo_pred(n):
    """The demo's choice (temp/n78_meter_breakdown.py): FFN down (ffn.2), attention output (o_proj), the memory block's
    big spike-input Linears (W_k / W_v / W_q / W_ff, 8192 -> 2048). Not qkv / FFN up / head / engram (inputs 79-88%
    nonzero: a dense read is cheaper there), not the memory block's low-rank gates (16 / 1 outputs)."""
    return n.endswith("ffn.2") or n.endswith("o_proj") or any(n.endswith(f".W_{k}") for k in ("k", "v", "q", "ff"))
