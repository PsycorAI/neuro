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

GATED variant (Gated DeltaNet, NVIDIA ICLR'25): per-timestep data-dependent forget
alpha_t in (0,1) replaces fixed lam. Recurrence:
    prevM = alpha[t] * M[t-1]    (was lam*M)
    everything else identical (k_hat normalize, beta-weighted error write, read).
Chunkwise: cumulative-alpha products gamma_t = prod_{s=1..t} alpha[s] replace lam^t.
All lam^(t-b) become gamma_t/gamma_b; lam^(t+1) becomes gamma_t (carry weight);
lam^C becomes gamma_C (chunk carry decay). Same triangular solve structure.

Identical math to the sequential loop (verified by temp/test_delta_chunked.py to <1e-4),
but O(T/C) kernel launches instead of O(T) — the prerequisite for delta at real-LM scale.

Public API:
    delta_chunked(v_seq, k_seq, q_seq, beta_seq, M_init, lam, chunk=64)
      v_seq, k_seq, q_seq: (B, T, D)   beta_seq: (B, T, 1)   M_init: (B, D, D)
      returns: r_seq (B, T, D), M_final (B, D, D)
    delta_chunked_gated(v_seq, k_seq, q_seq, beta_seq, alpha_seq, M_init, chunk=64)
      adds alpha_seq: (B, T, 1) data-dependent forget (replaces scalar lam).
    delta_sequential / delta_sequential_gated: reference O(T) loops (test oracles).
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


def delta_sequential_gated(v_seq, k_seq, q_seq, beta_seq, alpha_seq, M_init):
    """Reference O(T) loop — Gated DeltaNet: per-timestep alpha replaces scalar lam."""
    B, T, D = v_seq.shape
    M = M_init.clone()
    out = torch.empty_like(v_seq)
    for t in range(T):
        k = k_seq[:, t, :]
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        a = alpha_seq[:, t, :].unsqueeze(2)                        # (B,1,1)
        prevM = a * M
        Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)
        u = beta_seq[:, t, :] * (v_seq[:, t, :] - Mk)
        M = prevM + torch.bmm(u.unsqueeze(2), kn.unsqueeze(1))
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


def delta_chunked_gated(v_seq, k_seq, q_seq, beta_seq, alpha_seq, M_init, chunk=64):
    """Chunkwise-parallel GATED delta rule (Gated DeltaNet).

    Per-timestep data-dependent forget alpha_t in (0,1) replaces fixed lam.
    Within each chunk, build the cumulative-alpha product gamma_t = prod_{s<=t} alpha_s
    (relative to the chunk start), then every fixed-lam factor becomes a gamma ratio:
      lam^(t-b) -> gamma_t / gamma_b   (within-chunk decay between positions)
      lam^(t+1) -> gamma_t * alpha_0_carry  (carry weight)  -- here we fold the chunk
                                                                start into the carry alpha
      lam^C     -> gamma_C  (chunk-final decay applied to carry M)
    Numerical care: do all gamma in log-domain (cumsum log alpha), then exp differences.
    """
    B, T, D = v_seq.shape
    device = v_seq.device
    work = torch.float32
    V = v_seq.to(work)
    Q = q_seq.to(work)
    beta = beta_seq.to(work)
    a_seq = alpha_seq.to(work).clamp(1e-6, 1.0)      # (B,T,1)
    M = M_init.to(work)
    Kr = k_seq.to(work)
    Kn = Kr / (Kr.norm(dim=-1, keepdim=True) + 1e-6)
    log_a = torch.log(a_seq).squeeze(-1)             # (B,T)

    r_out = torch.empty(B, T, D, device=device, dtype=work)
    eye = torch.eye(chunk, device=device, dtype=work)

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        C = end - start
        Vc = V[:, start:end, :]
        Kc = Kn[:, start:end, :]
        Qc = Q[:, start:end, :]
        bc = beta[:, start:end, :]
        la = log_a[:, start:end]                     # (B,C)

        # cumulative log-decay within the chunk, INCLUSIVE: G[t] = log a[0]+...+a[t]
        G = la.cumsum(dim=1)                          # (B,C)
        # Within-chunk pairwise decay between positions: dec[t,b] = gamma_t/gamma_b
        #   for b<=t: exp(G[t] - G[b]); for b>t: garbage (masked out).
        # We need it in (B,C,C). G_t and G_b broadcast:
        Gt = G.unsqueeze(2)                           # (B,C,1)
        Gb = G.unsqueeze(1)                           # (B,1,C)
        dec_pair = torch.exp(Gt - Gb)                 # (B,C,C); entry [t,b]=gamma_t/gamma_b

        idx = torch.arange(C, device=device, dtype=work)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        strict = (diff > 0).to(work)
        incl = (diff >= 0).to(work)
        # Carry weight at position t: prod of alphas at 0..t  = gamma_t (inclusive)
        d_carry = torch.exp(G).unsqueeze(-1)          # (B,C,1)
        # Chunk-final decay applied to M_carry for the NEXT chunk: gamma_C = sum of all log alphas
        d_chunkC = torch.exp(G[:, -1])                # (B,)

        # --- solve for pseudo-values U ---
        # A[t,b] = (gamma_t/gamma_b) (k_b . k_t) for b<t
        KK = torch.bmm(Kc, Kc.transpose(1, 2))        # (B,C,C)
        A = KK * dec_pair * strict.unsqueeze(0)
        # Targ[t] = v[t] - gamma_t * (M_carry @ k_t)
        carry_k = torch.bmm(M, Kc.transpose(1, 2)).transpose(1, 2)
        Targ = Vc - d_carry * carry_k
        L = bc * A + eye[:C, :C].unsqueeze(0)
        RHS = bc * Targ
        U = torch.linalg.solve_triangular(L, RHS, upper=False, unitriangular=True)

        # --- read: r[t] = gamma_t M_carry@q_t + sum_{b<=t} (gamma_t/gamma_b)(k_b.q_t) u_b ---
        QK = torch.bmm(Qc, Kc.transpose(1, 2))
        Bmat = QK * dec_pair * incl.unsqueeze(0)
        r_within = torch.bmm(Bmat, U)
        cross = torch.bmm(M, Qc.transpose(1, 2)).transpose(1, 2) * d_carry
        r_out[:, start:end, :] = cross + r_within

        # --- carry update: M_next = gamma_C * M_carry + sum_b (gamma_C/gamma_b) u_b outer k_b ---
        # weights per b: w_b = gamma_C / gamma_b = exp(G[-1] - G[b])
        w = torch.exp(G[:, -1:].unsqueeze(-1) - G.unsqueeze(-1))   # (B,C,1)
        M = d_chunkC.view(B, 1, 1) * M + torch.bmm((U * w).transpose(1, 2), Kc)

    return r_out.to(v_seq.dtype), M.to(v_seq.dtype)
