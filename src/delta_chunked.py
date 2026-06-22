"""Chunkwise-parallel delta-rule (Gated DeltaNet) fast-weight memory.

Sequential recurrence (matches model.py SpikingHebbianBlock.step, scalar-lam path):
    k_hat = k / (||k|| + 1e-6)
    prevM = lam * M[t-1]
    u[t]  = beta[t] * (v[t] - prevM @ k_hat[t])      # error-driven pseudo-value
    M[t]  = prevM + u[t] outer k_hat[t]
    r[t]  = M[t] @ q[t]

The write `M[t] = lam*M[t-1] + u[t] outer k_hat[t]` is exactly Hebbian (eta=1) once
the pseudo-value u[t] is known. The only sequential coupling is that u[t] depends on
M[t-1] (hence on u[b<t]); within a chunk that is a unit-lower-triangular solve (the
DeltaNet WY / UT transform). After solving for U, the read and carry-update reuse the
same chunkwise matmuls as hebbian_chunked.

Identical math to the sequential loop (verified by temp/test_delta_chunked.py to <1e-4),
but O(T/C) kernel launches instead of O(T) — the prerequisite for delta at real-LM scale.

Public API:
    delta_chunked(v_seq, k_seq, q_seq, beta_seq, M_init, lam, chunk=64)
      v_seq, k_seq, q_seq: (B, T, D)   beta_seq: (B, T, 1)   M_init: (B, D, D)
      returns: r_seq (B, T, D), M_final (B, D, D)
    delta_sequential(...): same signature, reference O(T) loop (test oracle).
"""
import torch


def delta_sequential(v_seq, k_seq, q_seq, beta_seq, M_init, lam):
    """Reference O(T) loop — mirrors model.py step() delta_rule (scalar lam)."""
    B, T, D = v_seq.shape
    M = M_init.clone()
    out = torch.empty_like(v_seq)
    for t in range(T):
        k = k_seq[:, t, :]
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        prevM = lam * M
        Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)          # (B,D)
        u = beta_seq[:, t, :] * (v_seq[:, t, :] - Mk)              # (B,D)
        M = prevM + torch.bmm(u.unsqueeze(2), kn.unsqueeze(1))     # rank-1 write
        out[:, t, :] = torch.bmm(M, q_seq[:, t, :].unsqueeze(2)).squeeze(2)
    return out, M


def delta_chunked(v_seq, k_seq, q_seq, beta_seq, M_init, lam, chunk=64):
    """Chunkwise-parallel delta rule. See module docstring for the derivation."""
    B, T, D = v_seq.shape
    device = v_seq.device
    work = torch.float32                       # accumulate in fp32 even if bf16 in
    V = v_seq.to(work)
    Q = q_seq.to(work)
    beta = beta_seq.to(work)                   # (B,T,1)
    M = M_init.to(work)
    Kr = k_seq.to(work)
    Kn = Kr / (Kr.norm(dim=-1, keepdim=True) + 1e-6)   # normalized keys (B,T,D)

    r_out = torch.empty(B, T, D, device=device, dtype=work)
    eye = torch.eye(chunk, device=device, dtype=work)

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        C = end - start
        Vc = V[:, start:end, :]
        Kc = Kn[:, start:end, :]
        Qc = Q[:, start:end, :]
        bc = beta[:, start:end, :]             # (B,C,1)

        idx = torch.arange(C, device=device, dtype=work)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)        # (C,C) = t - b
        lam_pow = lam ** diff.clamp(min=0)                # lam^(t-b), garbage above diag
        strict = (diff > 0).to(work)                      # b < t
        incl = (diff >= 0).to(work)                       # b <= t
        d1 = (lam ** (idx + 1)).view(1, C, 1)             # lam^(t+1)

        # --- solve for pseudo-values U: (I + diag(beta) A) U = diag(beta) Targ ---
        # A[t,b] = lam^(t-b) (k_b . k_t) for b<t  (the lam*M[t-1]@k_t coupling)
        KK = torch.bmm(Kc, Kc.transpose(1, 2))            # (B,C,C): k_t . k_b at [t,b]
        A = KK * (lam_pow * strict).unsqueeze(0)
        # Targ[t] = v[t] - lam^(t+1) * (M_carry @ k_t)
        carry_k = torch.bmm(M, Kc.transpose(1, 2)).transpose(1, 2)   # (B,C,D): M@k_t
        Targ = Vc - d1 * carry_k
        L = bc * A + eye[:C, :C].unsqueeze(0)             # unit lower-triangular
        RHS = bc * Targ
        U = torch.linalg.solve_triangular(L, RHS, upper=False, unitriangular=True)

        # --- read: r[t] = lam^(t+1) M_carry@q_t + sum_{b<=t} lam^(t-b)(k_b.q_t) u_b ---
        QK = torch.bmm(Qc, Kc.transpose(1, 2))            # (B,C,C): q_t . k_b at [t,b]
        Bmat = QK * (lam_pow * incl).unsqueeze(0)         # inclusive (post-update read)
        r_within = torch.bmm(Bmat, U)
        cross = torch.bmm(M, Qc.transpose(1, 2)).transpose(1, 2) * d1   # (B,C,D)
        r_out[:, start:end, :] = cross + r_within

        # --- carry update: M = lam^C M_carry + sum_b lam^(C-1-b) u_b outer k_b ---
        w = (lam ** (C - 1 - idx)).view(1, C, 1)
        M = (lam ** C) * M + torch.bmm((U * w).transpose(1, 2), Kc)

    return r_out.to(v_seq.dtype), M.to(v_seq.dtype)
