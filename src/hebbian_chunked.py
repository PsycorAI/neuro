"""Chunkwise-parallel Hebbian fast-weight memory (TFLA-style).

Sequential recurrence:
    M[t] = lam * M[t-1] + eta * v[t] outer k[t]
    r[t] = M[t] @ q[t]

Chunkwise form (chunk size C): within each chunk we compute the within-chunk
contribution and the cross-chunk contribution from the prior `M_carry`. Both
collapse to matmuls so a chunk's work fuses into 3-4 large bmm kernels
(GPU-saturating) instead of C separate sequential bmms.

This is identical math to fla-org/flash-linear-attention's chunkwise kernel
for decay-linear-attention; we just write it in plain pytorch to avoid the
dependency. Numerically exact vs sequential (matmul associativity).

Public API:
    hebbian_chunked(v_seq, k_seq, q_seq, M_init, lam, eta, chunk=64)
      v_seq, k_seq, q_seq: (B, T, D)
      M_init: (B, D, D)
      returns: r_seq (B, T, D), M_final (B, D, D)
"""
import torch


def hebbian_chunked(v_seq, k_seq, q_seq, M_init, lam, eta, chunk=64):
    """Chunkwise-parallel Hebbian memory.

    M[t] = lam * M[t-1] + eta * v[t] outer k[t]
    r[t] = M[t] @ q[t]

    Within a chunk of length C:
      r_within[a] = sum_{b<=a} lam^(a-b) * eta * (k[b] . q[a]) * v[b]
                  = ((Q @ K^T) * lam_pow * eta * causal_mask) @ V       (matmul form)
      r_cross[a]  = lam^(a+1) * M_carry @ q[a]
      r[a] = r_cross[a] + r_within[a]

    Update for next chunk:
      M_next = lam^C * M_carry + eta * sum_{a=0..C-1} lam^(C-1-a) v[a] outer k[a]
             = lam^C * M_carry + eta * V^T @ diag(lam^(C-1-a)) @ K       (matmul form)
    """
    B, T, D = v_seq.shape
    device, dtype = v_seq.device, v_seq.dtype
    work = torch.float32                # accumulate in fp32 even if bf16 inputs
    v32 = v_seq.to(work)
    k32 = k_seq.to(work)
    q32 = q_seq.to(work)
    M = M_init.to(work)

    r_out = torch.empty(B, T, D, device=device, dtype=work)

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        C = end - start
        V = v32[:, start:end, :]                         # (B, C, D)
        K = k32[:, start:end, :]                         # (B, C, D)
        Q = q32[:, start:end, :]                         # (B, C, D)

        # decay matrices (C,C):  lam_pow[a,b] = lam^(a-b) if a>=b else 0
        idx = torch.arange(C, device=device, dtype=work)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)       # (C, C)
        causal = (diff >= 0).to(work)
        lam_pow = causal * (lam ** diff.clamp(min=0))    # zeros above diag

        # Within-chunk attention scores: (B, C_q, C_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) * lam_pow.unsqueeze(0) * eta
        r_within = torch.bmm(scores, V)                  # (B, C, D)

        # Cross-chunk read from carried M: r_cross[a] = lam^(a+1) * M @ q[a]
        # = (lam^(a+1) * Q[a]) @ M^T   (since M @ q == (q^T @ M^T)^T)
        # but M is (B, D, D) acting as M[d_out, d_k] and applied to q[d_k] -> r[d_out].
        # We had M defined as Hebbian write `M += v outer k`, so M[i,j] coefficients
        # for v_i k_j; r = M @ q means r[i] = sum_j M[i,j] q[j].
        a_idx = torch.arange(C, device=device, dtype=work)
        lam_a = (lam ** (a_idx + 1)).view(1, C, 1)        # (1, C, 1)
        # M @ Q^T per batch, then transpose: (B, D, D) @ (B, D, C) -> (B, D, C)
        cross = torch.bmm(M, Q.transpose(1, 2))            # (B, D, C)
        cross = cross.transpose(1, 2) * lam_a              # (B, C, D)

        r_out[:, start:end, :] = (cross + r_within).to(work)

        # Update M for next chunk:
        # M_next = lam^C * M + eta * sum_{a} lam^(C-1-a) v[a] outer k[a]
        #        = lam^C * M + eta * V^T_weighted @ K
        # weights w_a = lam^(C-1-a)
        w = (lam ** (C - 1 - a_idx)).view(1, C, 1)         # (1, C, 1)
        M = (lam ** C) * M + eta * torch.bmm(
            (V * w).transpose(1, 2), K)                    # (B, D, D)

    return r_out.to(dtype), M.to(dtype)
