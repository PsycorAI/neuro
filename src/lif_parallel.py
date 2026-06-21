"""Fixed-point Parallel Training (FPT) for the LIF cell.

Reference: arXiv:2506.12087 (June 2025).

The sequential LIF cell is:
    mem[t] = beta * mem[t-1] + cur[t] - spk_hard[t-1_carry]
    spk_hard[t] = Heaviside(beta * mem[t-1] + cur[t] - 1)

FPT observation: if we TREAT the spike sequence `s = spk_hard` as known, the LIF
membrane recurrence becomes purely linear in `cur` and `s`:
    mem[t] = beta * mem[t-1] + cur[t] - s[t]
        => mem[t] = sum_{i<=t} beta^(t-i) * (cur[i] - s[i])

That linear recurrence is a parallel scan (cumulative weighted sum), O(log T) work.

FPT iterates K times:
  1. Guess spike sequence s^(0) (e.g. all zeros).
  2. mem^(k) = parallel_scan(beta, cur - s^(k))
  3. s^(k+1) = Heaviside(mem^(k))   (because mem already has the -1 vth folded in
                                     via reset-by-subtraction; see below)
  4. Repeat. Typically converges in K=3 iterations.

This implementation parallelizes over the time dimension while keeping the
SAME math as the sequential `_lif_step` (vth=1, reset-by-subtraction,
atan surrogate gradient).

Public API:
    lif_parallel(cur, beta, vth=1.0, K=3, alpha=2.0)
        cur:  (B, T, N) input current
        returns: (spk (B, T, N) {0,1} with surrogate grad, mem_final (B, N))

The single-step result must match the sequential cell up to surrogate-gradient
order (forward should be exact when K is large enough).
"""
import math
import torch


@torch.no_grad()
def _scan_sequential(beta, x):
    """Reference O(T) cumulative-weighted-sum scan: y[t] = beta*y[t-1] + x[t].
    Used in tests, not in the fast path."""
    B, T, N = x.shape
    y = torch.zeros(B, N, device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    for t in range(T):
        y = beta * y + x[:, t, :]
        out[:, t, :] = y
    return out


def _scan_parallel(beta, x):
    """Parallel weighted prefix sum: y[t] = sum_{i<=t} beta^(t-i) * x[i].

    Closed form: y[t] = beta^t * cumsum_t( x[i] * beta^(-i) ).
    We compute via log-domain decay weights to avoid overflow for large T.

    Implementation: O(T) memory, O(T) wall (single cumsum kernel) -- but the
    cumsum kernel itself is GPU-parallel (log T span in practice).
    """
    B, T, N = x.shape
    device, dtype = x.device, x.dtype
    # decay_pow[t] = beta^t  (for t = 0..T-1)
    t_idx = torch.arange(T, device=device, dtype=dtype)
    log_beta = math.log(beta) if beta > 0 else -float("inf")
    # work in float32 for numerical safety even if x is bf16
    work_dtype = torch.float32
    x32 = x.to(work_dtype)
    # weighted: w[i] = beta^(-i) * x[i] ; we instead compute beta^t * cumsum(beta^(-i) x[i])
    # Use a numerically safe form:
    #   y[t] = sum_{i<=t} beta^(t-i) * x[i] = cumsum along T of (beta^(t-i) * x[i])
    # Rewrite as y[t] = beta * y[t-1] + x[t]; but to do parallel:
    # compute y = ifft-like via decaying basis? Simpler: use the closed form with
    # safe scaling.
    if beta == 0.0:
        return x32.to(dtype)
    # exponents (t - i): need triangular causal weights. We use the trick:
    #   y = beta^t * cumsum( beta^{-i} * x[i] )
    # For beta in (0,1), beta^{-i} explodes -> use log-domain via -t_idx*log(beta)
    # Then beta^t * cumsum -> cancellation. Safer alternative: compute in chunks.
    # For T<=2048 and beta>=0.9, beta^{-T} ~ 1/0.9^2048 ~ 10^94 -> overflow even in fp32.
    # So we use a numerically stable formulation by chunking with renormalization.
    return _scan_chunked(beta, x32).to(dtype)


def _scan_chunked(beta, x, chunk=512):
    """Numerically safe parallel scan via chunking with carry.

    Within each chunk we do a closed-form O(chunk^2) weighted sum (small, fast).
    Between chunks we propagate the final state. Wall is O(T) but each chunk's
    matmul fully utilizes the GPU. This is the FlashAttention-style trick
    applied to a 1D scan.
    """
    B, T, N = x.shape
    device, dtype = x.device, x.dtype
    out = torch.empty_like(x)
    # Pre-build the lower-triangular decay matrix for one chunk:
    # W[a,b] = beta^(a-b) if a>=b else 0 ; shape (chunk, chunk)
    # Then y_chunk = W @ x_chunk  (per (B,N) slot).
    # For the cross-chunk carry: y[start+a] += beta^(a+1) * carry_state
    carry = torch.zeros(B, N, device=device, dtype=dtype)
    last_chunk = None
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        L = end - start
        idx = torch.arange(L, device=device, dtype=dtype)
        # decay matrix W (L, L), lower-triangular
        a = idx.unsqueeze(1)  # (L,1)
        b = idx.unsqueeze(0)  # (1,L)
        diff = a - b
        W = torch.where(diff >= 0, beta ** diff, torch.zeros_like(diff))
        # within-chunk: y_chunk[t] = sum_{i in chunk, i<=t} beta^(t-i) * x[i]
        x_chunk = x[:, start:end, :]               # (B, L, N)
        y_chunk = torch.einsum("ab,bin->ain", W, x_chunk.transpose(0, 1)).transpose(0, 1)
        # add carry from previous chunks: each position t gets beta^(t+1) * carry
        carry_w = (beta ** (idx + 1)).view(1, L, 1)  # (1, L, 1)
        y_chunk = y_chunk + carry_w * carry.unsqueeze(1)
        out[:, start:end, :] = y_chunk
        # new carry is the final value of this chunk
        carry = y_chunk[:, -1, :]
    return out


def lif_parallel(cur, beta, vth=1.0, K=3, alpha=2.0):
    """FPT-style parallel LIF.

    Sequential reference:
        pre_mem[t]  = beta * post_mem[t-1] + cur[t]
        spk[t]      = Heaviside(pre_mem[t] - vth)
        post_mem[t] = pre_mem[t] - spk[t] * vth

    Substituting:
        pre_mem[t] = beta * pre_mem[t-1] + cur[t] - beta * vth * spk[t-1]
                   = scan(beta, x)[t]   with x[t] = cur[t] - beta * vth * spk_prev[t]

    where spk_prev = right-shift(spk) (spk_prev[0] = 0). So given a guess `s` for
    the spike sequence, we run the scan over `x = cur - beta*vth*shift(s)`, then
    re-derive s from `pre_mem > vth`. K=3-5 iterations converge.

    cur: (B, T, N) input current per timestep
    Returns: spk (B, T, N) in [0,1] with atan surrogate gradient, post_mem_final (B, N)
    """
    B, T, N = cur.shape
    s = torch.zeros_like(cur)
    with torch.no_grad():
        for _ in range(K):
            s_prev = torch.zeros_like(s)
            s_prev[:, 1:, :] = s[:, :-1, :]
            x = cur - beta * vth * s_prev
            pre_mem = _scan_chunked(beta, x)
            s = (pre_mem > vth).to(cur.dtype)
    # Final scan with the converged spike sequence (with grad)
    s_prev = torch.zeros_like(s)
    s_prev[:, 1:, :] = s[:, :-1, :]
    x = cur - beta * vth * s_prev
    pre_mem = _scan_chunked(beta, x)
    over = pre_mem - vth
    spk_hard = (over > 0).to(cur.dtype)
    g = (1.0 / (math.pi * alpha)) * torch.atan(math.pi * alpha * over)
    spk = spk_hard.detach() + g - g.detach()
    post_mem_final = pre_mem[:, -1, :] - spk_hard[:, -1, :] * vth
    return spk, post_mem_final
