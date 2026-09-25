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
import math
import torch


def unit_lower_inv(A_strict):
    """Inverse of (I + A_strict) for a strictly-lower-triangular A_strict (B,C,C),
    via Neumann-by-doubling: (I+A)^-1 = sum_k (-A)^k, exact since A is nilpotent
    (A^C = 0). Uses ceil(log2 C) batched matmuls — NO triangular solve, so the
    whole chunk kernel is torch.compile-friendly (unlike solve_triangular which
    graph-breaks). Compile-compatible replacement for solve_triangular."""
    B, C, _ = A_strict.shape
    N = -A_strict
    acc = torch.eye(C, device=A_strict.device, dtype=A_strict.dtype).expand(B, C, C).contiguous()
    Npow = N
    for _ in range(math.ceil(math.log2(C))):
        acc = acc + torch.bmm(Npow, acc)
        Npow = torch.bmm(Npow, Npow)
    return acc


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


def titans_sequential(v_seq, k_seq, q_seq, theta_seq, eta_seq, alpha_seq,
                      M_init, S_init):
    """Reference O(T) loop — Titans (Behrouz et al. 2025) neural-memory update.

    Memory M is updated by gradient descent on associative loss ℓ=‖Mk−v‖²
    (gradient ∝ (Mk−v)kᵀ), with MOMENTUM accumulating past surprise and a
    data-dependent forget gate:
        e_t = M_{t-1} @ k_hat_t  − v_t            (current surprise / gradient direction)
        S_t = η_t  S_{t-1}  − θ_t  e_t  k_hat_tᵀ  (momentum on surprise)
        M_t = (1−α_t) M_{t-1}  +  S_t             (forget then add momentum)
        r_t = M_t  q_t

    theta_seq, eta_seq, alpha_seq: (B, T, 1) — data-dependent gates.
    M_init, S_init: (B, D, D). Returns (out_seq, M_final, S_final).
    """
    B, T, D = v_seq.shape
    M = M_init.clone()
    S = S_init.clone()
    out = torch.empty_like(v_seq)
    for t in range(T):
        k = k_seq[:, t, :]
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        v = v_seq[:, t, :]
        q = q_seq[:, t, :]
        a = alpha_seq[:, t, :].unsqueeze(2)                    # (B,1,1)
        th = theta_seq[:, t, :].unsqueeze(2)
        et = eta_seq[:, t, :].unsqueeze(2)

        Mk = torch.bmm(M, kn.unsqueeze(2)).squeeze(2)          # M @ k̂  (B,D)
        e = Mk - v                                              # surprise (B,D)
        grad = torch.bmm(e.unsqueeze(2), kn.unsqueeze(1))      # (B,D,D)
        S = et * S - th * grad                                  # momentum
        M = (1 - a) * M + S
        out[:, t, :] = torch.bmm(M, q.unsqueeze(2)).squeeze(2)
    return out, M, S


def delta_fla_gated(v_seq, k_seq, q_seq, beta_seq, alpha_seq, M_init):
    """fla Triton chunk kernel wrapper — same signature/convention as
    delta_chunked_gated (k L2-normalized internally, raw q, per-step alpha gate,
    M laid out [value, key], read r = M @ q). Big speedup vs the eager chunk loop.

    Convention mapping to fla.chunk_gated_delta_rule:
      - k pre-normalized here (fla flag off) so q is NOT normalized (matches ours)
      - g = log(alpha)  (fla applies h *= exp(g))
      - beta passed in post-sigmoid space
      - scale=1.0 (no 1/sqrt(D) read scaling)
      - state transposed [v,k]<->[k,v] between our M and fla's h
    """
    # fla Triton kernel caps head_dim at 256. Our d_mem is typically 1024-2048,
    # so we split D into H heads of size <=256 for the kernel and concat back.
    # Since the delta rule is a rank-1 update in the (key,value) space, splitting
    # a single-head M of shape (D,D) into H diagonal blocks of (D/H, D/H) is a
    # LOSSY approximation for the model — but the model is trained with this
    # wrapper, so it just learns those diagonal-block dynamics from step 0.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    B, T, D = v_seq.shape
    MAX_HD = 256
    H = 1
    while D // H > MAX_HD:
        H *= 2
    assert D % H == 0, f"D={D} not divisible by required heads H={H}"
    HD = D // H
    dt = v_seq.dtype if v_seq.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    kn = k_seq / (k_seq.norm(dim=-1, keepdim=True) + 1e-6)
    g = torch.log(alpha_seq.clamp(1e-6, 1.0)).float().expand(B, T, H).contiguous()  # (B,T,H)
    g = g.unsqueeze(-1) if g.dim() == 2 else g
    beta_h = beta_seq.float().expand(B, T, H).contiguous().unsqueeze(-1) if beta_seq.dim() == 3 else beta_seq
    # (B,T,H,HD)
    q = q_seq.to(dt).view(B, T, H, HD)
    k = kn.to(dt).view(B, T, H, HD)
    v = v_seq.to(dt).view(B, T, H, HD)
    # initial_state: our M is (B,D,D); split into H diagonal blocks (B,H,HD,HD).
    # For H=1 this is the full M; for H>1 we lose off-block coupling (see note).
    if H == 1:
        h0 = M_init.transpose(-1, -2).unsqueeze(1).to(dt).contiguous()
    else:
        M_view = M_init.view(B, H, HD, H, HD)
        h0 = torch.diagonal(M_view, dim1=1, dim2=3).permute(0, 3, 1, 2)  # (B,H,HD,HD) [v,k]
        h0 = h0.transpose(-1, -2).to(dt).contiguous()                    # -> fla[k,v]
    o, hT = chunk_gated_delta_rule(
        q, k, v, g, beta_h.squeeze(-1),
        scale=1.0, initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False)
    r = o.reshape(B, T, D)                                          # concat heads
    if H == 1:
        M = hT.squeeze(1).transpose(-1, -2).float()
    else:
        # Reassemble diagonal-block M back to (B, D, D). Off-block regions = 0.
        hT_vk = hT.transpose(-1, -2).float()                        # (B,H,HD,HD) [v,k]
        M = torch.zeros(B, D, D, device=v_seq.device, dtype=torch.float32)
        for h in range(H):
            M[:, h*HD:(h+1)*HD, h*HD:(h+1)*HD] = hT_vk[:, h]
    return r, M


def delta_chunked_gated(v_seq, k_seq, q_seq, beta_seq, alpha_seq, M_init, chunk=64,
                        compile_inv=False):
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
        RHS = bc * Targ
        if compile_inv:
            # (I + bc*A)^-1 @ RHS via matmul-only doubling (compile-friendly)
            U = torch.bmm(unit_lower_inv(bc * A), RHS)
        else:
            L = bc * A + eye[:C, :C].unsqueeze(0)
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


def gdn2_sequential(v_seq, k_seq, q_seq, b_seq, w_seq, alpha_seq, M_init):
    """Reference O(T) loop — Gated DeltaNet-2 (arXiv:2605.22791) in our M=Sᵀ layout.

    Channel-wise erase gate b (d_k), channel-wise write gate w (d_v), channel-wise
    decay alpha (d_k). Per timestep, with kn = normalized key:
        M_dec = M * alpha                       # scale key-columns (decay)
        r_er  = M_dec @ (b ⊙ kn)                # read at gated key
        u     = (w ⊙ v) − r_er
        M     = M_dec + u ⊗ kn                   # write at plain key
        r     = M @ q
    Reduces to delta_sequential_gated when b≡w≡scalar β and alpha is scalar (the
    reduce-to-baseline check in temp/test_gdn2_osdn.py).

    b_seq: (B,T,d_k)  w_seq: (B,T,d_v)  alpha_seq: (B,T,d_k)  all in (0,1).
    """
    B, T, D = v_seq.shape
    M = M_init.clone()
    out = torch.empty_like(v_seq)
    for t in range(T):
        k = k_seq[:, t, :]
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        a = alpha_seq[:, t, :].unsqueeze(1)                       # (B,1,d_k)
        M_dec = M * a                                             # scale key cols
        bk = (b_seq[:, t, :] * kn).unsqueeze(2)                   # (B,d_k,1)
        r_er = torch.bmm(M_dec, bk).squeeze(2)                    # (B,d_v)
        u = w_seq[:, t, :] * v_seq[:, t, :] - r_er                # (B,d_v)
        M = M_dec + torch.bmm(u.unsqueeze(2), kn.unsqueeze(1))    # write at kn
        out[:, t, :] = torch.bmm(M, q_seq[:, t, :].unsqueeze(2)).squeeze(2)
    return out, M


def _gdn2_dense_chunk(S, kn, v, q, b, LogA, sub=16):
    """One chunk of gdn2_chunked_dense (pure function of tensor args, safe
    checkpoint target). Two-level GLA-style gating: intra-sub-block decay
    ratios exp(cs_t - cs_s) computed directly per (t,s,channel) pair (bounded
    <= 1 after clamp(max=0), which only touches the masked upper triangle);
    inter-sub-block ratios anchored at each row-block entry a_i, giving two
    factors exp(cs_t - a_i) and exp(a_i - cs_s) that are BOTH <= 1 by
    construction. No unbounded exponentials anywhere -- this replaced a
    midpoint-shift factorization that overflowed/zeroed real contributions
    for trained fast-decay channels (alpha ~ 1e-6/step). Intra-block terms
    are vectorized across sub-blocks in one einsum; requires C % sub == 0
    (caller pads chunking so this holds or falls to the ragged path)."""
    B, C, K = kn.shape
    pad = (-C) % sub
    if pad and C > sub:
        # Ragged tail longer than one sub-block (e.g. SFT/OPD batches padded to
        # their longest example, T % chunk up to chunk-1). The ragged path below
        # materializes (B,C,C,K) fp32 -- ~1.3GB/call at C=255,K=1280 -- which
        # OOM'd v3 SFT on the 5090 (2026-09-25). Zero-pad to a multiple of `sub`
        # instead: kn=v=q=b=0 and LogA=0 (alpha=1) rows give zero A/Bm rows+cols,
        # zero U, zero k_carry and leave cs[-1] unchanged -> exact no-op.
        z = lambda t: torch.nn.functional.pad(t, (0, 0, 0, pad))
        S, r = _gdn2_dense_chunk(S, z(kn), z(v), z(q), z(b), z(LogA), sub)
        return S, r[:, :C]
    work = kn.dtype
    dev = kn.device
    cs = torch.cumsum(LogA, dim=1)                                 # (B,C,K), <= 0
    Gt = torch.exp(cs)                                             # Gamma_t <= 1
    bk = b * kn

    if C % sub == 0:
        n_sub = C // sub
        csb = cs.view(B, n_sub, sub, K)
        d = (csb.unsqueeze(3) - csb.unsqueeze(2)).clamp(max=0.0)   # (B,n,sub,sub,K)
        Wp = torch.exp(d)
        bkb = bk.view(B, n_sub, sub, K)
        knb = kn.view(B, n_sub, sub, K)
        qb = q.view(B, n_sub, sub, K)
        A_int = torch.einsum('bntsk,bntk,bnsk->bnts', Wp, bkb, knb)
        B_int = torch.einsum('bntsk,bntk,bnsk->bnts', Wp, qb, knb)
        A = kn.new_zeros(B, C, C)
        Bm = kn.new_zeros(B, C, C)
        for i in range(n_sub):
            lo, hi = i * sub, (i + 1) * sub
            A[:, lo:hi, lo:hi] = A_int[:, i]
            Bm[:, lo:hi, lo:hi] = B_int[:, i]
            if lo > 0:
                a_i = cs[:, lo - 1:lo]                             # (B,1,K)
                lam = torch.exp(cs[:, lo:hi] - a_i)                # <= 1
                rs = kn[:, :lo] * torch.exp(a_i - cs[:, :lo])      # <= kn
                A[:, lo:hi, :lo] = torch.bmm(bk[:, lo:hi] * lam, rs.transpose(1, 2))
                Bm[:, lo:hi, :lo] = torch.bmm(q[:, lo:hi] * lam, rs.transpose(1, 2))
    else:
        # ragged tail chunk: direct per-pair (C small here)
        d = (cs.unsqueeze(2) - cs.unsqueeze(1)).clamp(max=0.0)
        Wp = torch.exp(d)
        A = torch.einsum('btsk,btk,bsk->bts', Wp, bk, kn)
        Bm = torch.einsum('btsk,btk,bsk->bts', Wp, q, kn)

    idx = torch.arange(C, device=dev)
    A = A * (idx.unsqueeze(1) > idx.unsqueeze(0)).to(work).unsqueeze(0)
    Bm = Bm * (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(work).unsqueeze(0)

    # Stable forward-substitution solve (matches the production scalar path's
    # default). Do NOT use unit_lower_inv here: with trained correlated keys
    # (MQAR-aug repeats keys, inner products -> 1) the Neumann-doubling
    # intermediates A^(2^j) grow to ~1e10+ before cancelling, and fp32
    # round-off of those intermediates blew up real training to NaN within
    # 300 steps (2026-07-15) while random-input parity stayed at 1e-6.
    eye = torch.eye(C, device=dev, dtype=work).unsqueeze(0)
    RHS = v - torch.bmm(Gt * bk, S)                                # v is w*v pre-scaled
    U = torch.linalg.solve_triangular(A + eye, RHS, upper=False,
                                      unitriangular=True)          # (B,C,V)
    r = torch.bmm(Gt * q, S) + torch.bmm(Bm, U)
    gC = cs[:, -1:]                                                # (B,1,K) log Gamma_C
    k_carry = torch.exp(gC - cs) * kn                              # (Gamma_C/Gamma_s)*kn <= kn
    S = torch.exp(gC).transpose(1, 2) * S + torch.bmm(k_carry.transpose(1, 2), U)
    return S, r


def gdn2_chunked_dense(v_seq, k_seq, q_seq, b_seq, w_seq, alpha_seq, M_init,
                       chunk=64, ckpt_chunks=False):
    """Chunkwise-parallel GDN-2 with the FULL DENSE state -- no head splitting.

    Exact WY-form reassociation of gdn2_sequential's recurrence (same math,
    different float association), expressed as batched GEMMs so the dense
    (d_mem x d_mem) state never needs to fit a kernel's SRAM tile -- this is
    what lifts the fla kernel's 256 key-headdim cap (whose forced block-
    diagonal split collapsed MQAR recall at H>=8; journey/memory 2026-07-15).

    Derivation (S = M^T in (K,V) layout, Gamma_t = per-channel cumprod of
    alpha within the chunk, kn = normalized key):
        u_t  = w_t*v_t - (Gamma_t*b_t*kn_t)^T S_0 - sum_{s<t} A[t,s] u_s
        A[t,s] = (b_t*kn_t) . (kn_s) weighted per-channel by Gamma_t/Gamma_s
        r_t  = S_0^T (Gamma_t*q_t) + sum_{s<=t} B[t,s] u_s
        S_C  = Gamma_C * S_0 + sum_s ((Gamma_C/Gamma_s)*kn_s) u_s^T
    The Gamma_t/Gamma_s pairings use GLA-style two-level chunking (see
    _gdn2_dense_chunk) so every exponential is <= 1 -- exact and overflow-free
    even for trained fast-decay channels. Same factorization family as fla's
    wy_fast.py, without its 256 key-headdim SRAM cap.

    Verified vs gdn2_sequential by temp/test_gdn2_chunked_dense.py (fwd+bwd,
    incl. alpha down to 1e-6 and mixed dead/live channels). Signature and
    convention identical to gdn2_sequential (M laid out [value,key], k
    L2-normalized internally, raw q, r = M @ q). Chunks are grad-checkpointed
    during training (house rule for sequential state loops).
    """
    from torch.utils.checkpoint import checkpoint as _ckpt
    B, T, D = v_seq.shape
    work = torch.float32
    # Hard-disable autocast for the whole chunked computation: bmm/einsum are
    # autocast-eligible, so an outer bf16 autocast would silently downcast the
    # "fp32-internal" GEMMs -- the triangular inverse is not bf16-stable and
    # NaN'd real training within 100 steps (2026-07-15). With autocast off,
    # production math is bit-identical to the parity-tested fp32 path.
    ac_off = torch.autocast(device_type=v_seq.device.type, enabled=False)
    with ac_off:
        Q = q_seq.to(work)
        Bg = b_seq.to(work)
        WV = (w_seq * v_seq).to(work)                 # write gate folded into v
        Kr = k_seq.to(work)
        Kn = Kr / (Kr.norm(dim=-1, keepdim=True) + 1e-6)
        LogA = torch.log(alpha_seq.to(work).clamp(1e-6, 1.0))      # (B,T,K) <= 0
        S = M_init.to(work).transpose(1, 2).contiguous()           # (B,K,V) = M^T
        r_parts = []
        # checkpoint is opt-in: the per-chunk tape is modest (~100MB/chunk at
        # B=24 D=1024) and recompute costs ~40% -- flip on only if VRAM-bound.
        grad_on = ckpt_chunks and torch.is_grad_enabled() and (
            v_seq.requires_grad or k_seq.requires_grad or S.requires_grad
            or q_seq.requires_grad or b_seq.requires_grad or w_seq.requires_grad
            or alpha_seq.requires_grad)
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            args = (S, Kn[:, start:end], WV[:, start:end], Q[:, start:end],
                    Bg[:, start:end], LogA[:, start:end])
            if grad_on:
                S, r = _ckpt(_gdn2_dense_chunk, *args, use_reentrant=False)
            else:
                S, r = _gdn2_dense_chunk(*args)
            r_parts.append(r)
        r_out = torch.cat(r_parts, dim=1)
    return r_out.to(v_seq.dtype), S.transpose(1, 2).to(M_init.dtype)


def gdn2_fla_gated(v_seq, k_seq, q_seq, b_seq, w_seq, alpha_seq, M_init, min_heads=1):
    """fla Triton chunk kernel wrapper for GDN-2 — same convention as
    gdn2_sequential (k L2-normalized internally, raw q, M laid out [value,
    key], read r = M @ q). Parity-validated against gdn2_sequential (see
    temp/test_gdn2_fla_parity.py): fwd max err ~1e-2 (fp32 float-order
    accumulation over T), bwd bf16 loss/grad rel err ~1%.

    Convention mapping to fla.ops.gdn2.chunk_gdn2 (mirrors delta_fla_gated):
      - k pre-normalized here (fla flag off) so q is NOT normalized (matches ours)
      - g = log(alpha)  (fla applies h *= exp(g)); b, w passed as channel-wise
        gates directly (GDN-2 already has vector erase/write gates, unlike
        the scalar-beta plain delta path)
      - scale=1.0 (no 1/sqrt(D) read scaling)
      - state transposed [v,k]<->[k,v] between our M and fla's h
      - same head_dim<=256 block-diagonal split as delta_fla_gated for
        d_mem > 256 (lossy off-block coupling, trained-through)
    """
    from fla.ops.gdn2 import chunk_gdn2
    B, T, D = v_seq.shape
    MAX_HD = 256
    H = max(1, min_heads)
    while D // H > MAX_HD:
        H *= 2
    assert D % H == 0, f"D={D} not divisible by required heads H={H}"
    HD = D // H
    dt = v_seq.dtype if v_seq.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    kn = k_seq / (k_seq.norm(dim=-1, keepdim=True) + 1e-6)
    g = torch.log(alpha_seq.clamp(1e-6, 1.0)).float()             # (B,T,D)
    q = q_seq.to(dt).view(B, T, H, HD)
    k = kn.to(dt).view(B, T, H, HD)
    v = v_seq.to(dt).view(B, T, H, HD)
    g = g.view(B, T, H, HD)
    b = b_seq.to(dt).view(B, T, H, HD)
    w = w_seq.to(dt).view(B, T, H, HD)
    # gdn2 kernel hard-asserts initial_state.dtype == float32 (the sibling
    # gated_delta_rule kernel accepts bf16; this one does not) -- keep h0 fp32.
    if H == 1:
        h0 = M_init.transpose(-1, -2).unsqueeze(1).float().contiguous()
    else:
        M_view = M_init.view(B, H, HD, H, HD)
        h0 = torch.diagonal(M_view, dim1=1, dim2=3).permute(0, 3, 1, 2)  # (B,H,HD,HD) [v,k]
        h0 = h0.transpose(-1, -2).float().contiguous()                   # -> fla[k,v]
    o, hT = chunk_gdn2(
        q, k, v, g, b, w,
        scale=1.0, initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=False)
    r = o.reshape(B, T, D)
    if H == 1:
        M = hT.squeeze(1).transpose(-1, -2).float()
    else:
        hT_vk = hT.transpose(-1, -2).float()
        M = torch.zeros(B, D, D, device=v_seq.device, dtype=torch.float32)
        for h in range(H):
            M[:, h*HD:(h+1)*HD, h*HD:(h+1)*HD] = hT_vk[:, h]
    return r, M


def gdn2_loop_multihead(v_seq, k_seq, q_seq, b_seq, w_seq, alpha_seq, M_init, chunk=64):
    """Diagnostic twin of gdn2_fla_gated: IDENTICAL head-splitting convention
    (same H-selection via the 256 cap, same block-diagonal M layout, same
    input/output reshape), but runs model._gdn2_chunk_fwd's eager reference
    loop per head-batch instead of the Triton kernel. Isolates 'is the recall
    regression a kernel-numerics issue or inherent to head-splitting itself'
    -- run both on the SAME trained weights and compare.
    """
    import sys
    from torch.utils.checkpoint import checkpoint
    model_mod = sys.modules["model"]
    _gdn2_chunk_fwd = model_mod._gdn2_chunk_fwd
    B, T, D = v_seq.shape
    MAX_HD = 256
    H = 1
    while D // H > MAX_HD:
        H *= 2
    assert D % H == 0, f"D={D} not divisible by required heads H={H}"
    HD = D // H
    kn = k_seq / (k_seq.norm(dim=-1, keepdim=True) + 1e-6)

    def to_heads(t):
        return t.view(B, T, H, HD).permute(0, 2, 1, 3).reshape(B * H, T, HD)
    q_h, k_h, v_h = to_heads(q_seq), to_heads(kn), to_heads(v_seq)
    b_h, w_h, a_h = to_heads(b_seq), to_heads(w_seq), to_heads(alpha_seq)

    if H == 1:
        M_h = M_init.clone()
    else:
        M_view = M_init.view(B, H, HD, H, HD)
        M_h = torch.diagonal(M_view, dim1=1, dim2=3).permute(0, 3, 1, 2).reshape(B * H, HD, HD).contiguous()

    r_chunks = []
    training = torch.is_grad_enabled()
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        args = (M_h, k_h[:, s:e], v_h[:, s:e], q_h[:, s:e], b_h[:, s:e], w_h[:, s:e], a_h[:, s:e])
        if training:
            M_h, r_ch = checkpoint(_gdn2_chunk_fwd, *args, use_reentrant=False)
        else:
            M_h, r_ch = _gdn2_chunk_fwd(*args)
        r_chunks.append(r_ch)
    r_h = torch.cat(r_chunks, dim=1)                                   # (B*H,T,HD)

    r = r_h.reshape(B, H, T, HD).permute(0, 2, 1, 3).reshape(B, T, D)
    if H == 1:
        M = M_h
    else:
        M_hd = M_h.reshape(B, H, HD, HD)
        M = torch.zeros(B, D, D, device=v_seq.device, dtype=M_init.dtype)
        for h in range(H):
            M[:, h*HD:(h+1)*HD, h*HD:(h+1)*HD] = M_hd[:, h]
    return r, M


def osdn_sequential(v_seq, k_seq, q_seq, beta_seq, alpha_seq, M_init,
                    eta=0.003, d_min=0.5, d_max=2.0, eps=1e-6):
    """Reference O(T) loop — OSDN online-preconditioned delta rule (arXiv:2605.13473).

    Diagonal preconditioner d (d_k) adapts online; error read on the plain key,
    rank-1 write on the preconditioned key k̃ = d ⊙ kn. Per timestep:
        prevM = alpha * M
        u     = beta * (v − prevM @ kn)          # read plain key
        M     = prevM + u ⊗ (d ⊙ kn)             # write preconditioned key
        r     = M @ q
        d     = clamp(d + (eta·beta/‖k‖²)(1 − beta⟨d,k²⟩)·k², d_min, d_max)
    d init = 1 (identity preconditioner ⇒ exact delta rule at t=0). With d≡1 held
    fixed this reduces to delta_sequential_gated (the baseline check).

    alpha_seq: (B,T,1) data-dependent forget (pass ones*lam for scalar decay).
    """
    B, T, D = v_seq.shape
    M = M_init.clone()
    d = torch.ones(B, k_seq.size(-1), device=v_seq.device, dtype=v_seq.dtype)
    out = torch.empty_like(v_seq)
    for t in range(T):
        k = k_seq[:, t, :]
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        beta = beta_seq[:, t, :]                                  # (B,1)
        a = alpha_seq[:, t, :].unsqueeze(2)                       # (B,1,1)
        prevM = a * M
        Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)         # read plain key
        u = beta * (v_seq[:, t, :] - Mk)                          # (B,d_v)
        k_tilde = d * kn                                          # preconditioned
        M = prevM + torch.bmm(u.unsqueeze(2), k_tilde.unsqueeze(1))
        out[:, t, :] = torch.bmm(M, q_seq[:, t, :].unsqueeze(2)).squeeze(2)
        # online preconditioner update (elementwise on key dim)
        k2 = kn * kn                                              # (B,d_k)
        nrm = k2.sum(dim=-1, keepdim=True).clamp(min=eps)         # ‖k‖²
        dot = (d * k2).sum(dim=-1, keepdim=True)                  # ⟨d,k²⟩
        d = (d + (eta * beta / nrm) * (1 - beta * dot) * k2).clamp(d_min, d_max)
    return out, M
