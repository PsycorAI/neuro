# Custom Triton delta_chunked_gated kernel — scoping document

## Why we're writing our own, not using fla

Attempted `fla 0.5.1` `chunk_gated_delta_rule` on RTX 5080 (Blackwell, sm_120):

- **Forward + backward RUN** (no crash — the prior session's crash-level Blackwell bug appears fixed)
- **BUT: outputs are wildly wrong.** Compared to fla's own naive reference:

  | T | has NaN/Inf | max diff | p95 diff | median diff |
  |---|---|---|---|---|
  | 32 | no | 1.6e12 | 6.0e10 | 9.5e3 |
  | 64 | no | 2.3e24 | 5.0e22 | 4.1e12 |
  | 128 | yes | NaN | NaN | NaN |
  | 256 | yes | NaN | NaN | NaN |

Correct output should have median diff near machine epsilon. This is semantic failure of fla's chunk kernel on Blackwell. **Unusable for us.**

Filed as a data point for fla team; not our bug to fix.

## Target: custom Triton kernel matching `src/delta_chunked.py:delta_chunked_gated`

Our reference math (single head, matches what we already ship):

```
Within a chunk of length C=64:
    k_hat[t]   = k[t] / (||k[t]|| + eps)
    G[t]       = sum_{s=0..t} log(alpha[s])            # cumulative log-decay
    dec[t,b]   = exp(G[t] - G[b])                       # within-chunk pairwise decay

    A[t,b]     = dec[t,b] * <k_hat[b], k_hat[t]> * beta[t]       for b<t, else 0
    L          = I + A                                              # unit-lower-tri
    Targ[t]    = v[t] - exp(G[t]) * (M_carry @ k_hat[t])
    U          = solve_triangular(L, beta*Targ)                    # pseudo-values

    QK[t,b]    = <q[t], k_hat[b]> * dec[t,b] * [b<=t]
    r[t]       = exp(G[t]) * (M_carry @ q[t]) + sum_{b<=t} QK[t,b] * U[b]

Carry:
    M_next     = gamma_C * M_carry + sum_b (gamma_C/gamma_b) U[b] outer k_hat[b]
    (gamma_C = exp(G[C-1]))
```

## Kernel design

### Forward — one Triton kernel per (batch, chunk)

Grid: `(B, T//C)`. Each program:
1. Load `V, K, Q, beta, alpha` chunk `[C, D]` (D=d_mem, C=64) — bf16 to fp32
2. Load `M_carry` — `[D, D]` — fp32 (from previous chunk's output)
3. Compute in registers/SMEM:
   - `k_hat`  (element-wise normalize)
   - `log_alpha, G` (cumsum)
   - `KK = k_hat @ k_hat.T` (C×C bmm, small)
   - `A = KK * dec_pair * strict * beta` (elementwise on C×C)
   - `Targ = V - exp(G) * (M_carry @ k_hat.T)` (C×D outputs)
   - Triangular solve for `U` — this is the tricky part, see below
4. Emit output `r = exp(G) * (M_carry @ q.T) + (QK * dec_pair * incl) @ U`
5. Update `M_carry` for next chunk (rank-C update)

### The triangular solve

Triton has no built-in triangular solve. Options:

**Option A: Substitution loop.** For each row t=0..C-1:
```
U[t] = beta[t] * Targ[t] - sum_{b<t} L[t,b] * U[b]
```
Naive is O(C^2 * D) work per chunk, but stays in registers. Not vectorized well.

**Option B: WY / UT transform.** DeltaNet paper's factorization: `L = I - beta * K_tri @ K.T`, then invert via `L^{-1} = I + beta * K @ (I - beta * K.T @ K)^{-1} @ K.T`. Nested C×C solve. Works, but nesting doesn't help.

**Option C: Materialize L^{-1} once.** For C=64 the C×C matrix is 16KB fp32 — fits in SMEM. Use built-in `tl.dot` for the pseudo-values compute `U = L_inv @ (beta * Targ)`. Requires precomputed L_inv.

**Recommendation: option C.** Compute L_inv on the host (Python) OR in a preliminary pass per chunk. Since L is upper-triangular in one convention, invert via block back-substitution — one Triton kernel that computes `L_inv[C, C]` from `L[C, C]` (~1μs per launch). Then main kernel uses `tl.dot` for the solve step.

Actually simpler: for C=64, do the substitution inline in the main kernel — 64 sequential steps, each is a `tl.sum(L[t, :t] * U[:t])`. Slow but correct. Given C^2=4096 ops per position, and D=2048 matmul cost per position, the substitution overhead is bounded.

### Backward

Follows the standard chunkwise DeltaNet backward (see fla's `chunk.py:120 chunk_gated_delta_rule_bwd` as reference algorithm even though their kernel is buggy). Key gradients:

- `dU = dr @ QK.T + rank-1 from carry gradient` — feeds back through the triangular solve
- `dTarg = solve_triangular_T(L, dU)` — transposed solve
- `dV = beta * dTarg`
- `dK, dQ` — matmul contributions
- `dg (log alpha)` — cumulative gradient through the exp(G) terms

Backward is where our own kernel gets HARD. Est: 3-5 dedicated dev days for a numerically-verified, autograd-compatible backward.

## Estimated speedup

vs current eager `delta_chunked_gated`:

- Eliminates ~8 bmm launches per chunk → ~5-10× reduction in kernel launch overhead
- Fuses SGEMM chains → ~2× memory-bandwidth win on activations
- **Realistic net: 2-4× throughput on the delta pathway** in isolation, ~1.5-2× total training throughput at 500M (since delta is ~50-60% of student time under KD)

At 5090's current 4.1k tok/s: post-Triton estimate 6-8k tok/s → 3B tokens goes from 8 days to 5 days.

## Realistic dev budget

**5 focused days** minimum, phased:
1. Day 1: forward kernel skeleton + reference-match on B=1, small T
2. Day 2: forward optimization (register/SMEM tiling), scale to full B, T
3. Day 3: backward kernel — dV, dK, dQ paths
4. Day 4: backward — dU (through triangular solve), dg, dbeta
5. Day 5: end-to-end training smoke, autograd numerics vs eager kernel

## Fallback / interim

If a 5-day sprint isn't feasible:

- **Eager kernel optimization**: torch.compile is currently disabled because of the delta kernel. Investigate WHY — if it's a specific op (linalg.solve_triangular), we may be able to compile everything EXCEPT that one op. `@torch.compiler.disable` on the delta_chunked_gated function was tried in earlier session and gave ~14% (see `reference_torch_compile_delta.md`). Revisit with fresh torch — the compile situation improved throughout 2025.
- **Batch the bmm ops** in eager: `torch.bmm` calls in delta_chunked can be combined into single-launch mega-bmms for K,V,Q simultaneously. Small win.

## Next session

If the user wants to pursue the Triton rewrite:
1. Start with forward kernel prototype in `src/delta_chunked_triton.py`
2. Compare numerics to `delta_chunked_gated` at C=8, D=32, B=1 first
3. Scale up once forward is correct
4. Only then attempt backward
