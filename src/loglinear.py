"""Log-linear GDN-2 memory (Stage-A de-risking spike).

arXiv:2506.04761 (Log-Linear Attention) composed with our GDN-2 memory rule,
per docs/loglinear_attention_design.md §5 Stage A. **Scope: spike only.** This
module is intentionally NOT imported by the model or any trainer yet -- it is
developed and parity-tested in isolation so the live `delta_chunked` path
(imported by running trainers) stays untouched until this is proven. Fold into
delta_chunked.py only after fwd+bwd parity + a 500-step NaN watch pass.

Idea. Plain GDN-2 keeps ONE decayed state M_t that sums every past write; the
single fixed-size state is the associative-recall ceiling this project has shown
is architectural (4x budget didn't move it). Log-linear keeps the SAME writes but
decomposes the causal prefix [0,t] into O(log t) Fenwick (dyadic) blocks, each
with its own accumulated sub-state, and reads them back with DATA-DEPENDENT
per-level scalar weights lw_t^(l):

    r_t = base_w_t * (M_init_carry @ q_t)  +  sum_{l active} lw_t^(l) * (block_l @ q_t)

Semantics (Option A in the design doc): the WRITE path is exactly GDN-2 -- the
delta/erase pseudo-value u_t is computed against the FULL state (= base + sum of
all live blocks), so nothing about capacity or the delta correction changes.
Log-linear ONLY reweights read contributions by age. Therefore, with every
lw_t^(l) == 1 (and base_w == 1), the weighted sum telescopes back to plain GDN-2
exactly -- the parity anchor the chunked kernel is tested against.

The Fenwick tower is a binary counter: after t writes, the occupied slots are
exactly the set bits of t, slot l holding a contiguous decayed block of 2^l
writes. Merging two level-l blocks (both on the same per-step decay clock) is an
exact add. This gives the O(log t) disjoint cover the readout reweights.

`gdn2_loglinear_sequential` below is the O(T log T) reference oracle: it defines
the semantics unambiguously. The chunked parallel kernel (to follow) must match
it to ~1e-4 fwd+bwd, including trained fast-decay channels (alpha ~ 1e-6/step),
which is the numerically dangerous part (more decay factors compose across
levels than in the single-state path).
"""
import torch


def gdn2_loglinear_sequential(v_seq, k_seq, q_seq, b_seq, w_seq, alpha_seq,
                              lw_seq, M_init, base_weight=None, blk=1):
    """O(T log T) reference oracle for log-linear GDN-2 (the semantics spec).

    Convention identical to delta_chunked.gdn2_sequential (M laid out
    [value, key], key L2-normalized internally, raw q, read r = M @ q), plus:

    lw_seq: (B, T, L) data-dependent per-level read weights. lw_seq[:, t, l]
            scales the level-l Fenwick block of COMPLETED chunks when reading at
            time t. L is the number of tower levels (must satisfy 2^L*blk > T).
            All-ones reduces this EXACTLY (fp32 float-association ~1e-5) to
            gdn2_sequential -- the parity anchor, at any blk.
    base_weight: (B, T) weight on the persistent M_init carry block, or None
            (treated as 1 -> also part of the reduce-to-GDN2 anchor).
    M_init: (B, d_v, d_k) pre-sequence state; the always-live base block.
    blk: Fenwick unit size in tokens. The CURRENT (incomplete) chunk of `blk`
         tokens is read densely with weight 1 (`pending`); only COMPLETED chunks
         enter the reweighted tower. blk=1 => token-level; blk=C => the
         chunk-level semantics `gdn2_loglinear_chunked` (chunk=C) must match for
         general lw. The paper's kernel is chunk-level.

    Returns (out (B,T,d_v), slots (list[L] or None), base, pending).
    Final full state = base + pending + sum(s for s in slots if s is not None).
    """
    B, T, D = v_seq.shape
    L = lw_seq.shape[-1]
    # binary-counter Fenwick tower over COMPLETED chunks: slot[l] = decayed
    # level-l block (2^l chunks), or None. `pending` accumulates the current
    # incomplete chunk's writes (read densely, weight 1).
    slot = [None] * L
    base = M_init.clone()
    pending = torch.zeros_like(M_init)
    filled = 0
    out = torch.empty_like(v_seq)

    for t in range(T):
        k = k_seq[:, t, :]
        kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        a = alpha_seq[:, t, :].unsqueeze(1)                    # (B,1,d_k) decay on key cols

        # 1. decay every live block == decaying the whole accumulated state.
        base = base * a
        pending = pending * a
        for l in range(L):
            if slot[l] is not None:
                slot[l] = slot[l] * a

        # 2. GDN-2 write: pseudo-value u computed against the FULL decayed state
        #    (base + pending + all live blocks) -- delta/erase correction unchanged.
        M_dec = base + pending
        for l in range(L):
            if slot[l] is not None:
                M_dec = M_dec + slot[l]
        bk = (b_seq[:, t, :] * kn).unsqueeze(2)                # (B,d_k,1)
        r_er = torch.bmm(M_dec, bk).squeeze(2)                 # (B,d_v)
        u = w_seq[:, t, :] * v_seq[:, t, :] - r_er             # (B,d_v)
        write = torch.bmm(u.unsqueeze(2), kn.unsqueeze(1))     # (B,d_v,d_k) = u (x) kn
        pending = pending + write
        filled += 1

        # 3. log-linear read: pending (weight 1) + per-level weighted tower + base.
        q = q_seq[:, t, :].unsqueeze(2)                        # (B,d_k,1)
        bw = q.new_ones(B, 1) if base_weight is None else base_weight[:, t].view(B, 1)
        r = bw * torch.bmm(base, q).squeeze(2) + torch.bmm(pending, q).squeeze(2)
        for l in range(L):
            if slot[l] is not None:
                r = r + lw_seq[:, t, l].view(B, 1) * torch.bmm(slot[l], q).squeeze(2)
        out[:, t, :] = r

        # 4. at a chunk boundary, push pending into the tower (binary carry = add).
        if filled == blk:
            carry = pending
            l = 0
            while carry is not None:
                if l >= L:
                    raise ValueError(f"L={L} too small for T={T}, blk={blk}")
                if slot[l] is None:
                    slot[l] = carry
                    carry = None
                else:
                    carry = slot[l] + carry
                    slot[l] = None
                    l += 1
            pending = torch.zeros_like(M_init)
            filled = 0

    return out, slot, base, pending


def _gdn2_ll_chunk(S_full, kn, v, q, b, LogA, sub=16):
    """Log-linear chunk primitive. Reproduces the intra-chunk WY math of
    delta_chunked._gdn2_dense_chunk (two-level GLA factorization so every
    exponential is <= 1, overflow-free for fast-decay channels) but returns the
    pieces the log-linear wrapper needs instead of folding in the single-state
    read/update:
      U    (B,C,V) pseudo-values (solved against the FULL state S_full)
      r_in (B,C,V) intra-chunk read = Bm @ U (the current chunk, weight 1)
      Gtq  (B,C,K) Gamma_t * q -- applying (Gtq @ block) reads a tower block
                   decayed to row t (equals reading block-decayed-to-t by q)
      dS   (B,K,V) this chunk's state contribution, referenced to chunk END
      gC   (B,1,K) log Gamma_C (chunk-final decay, to age the tower each chunk)
    S_full is (B,K,V) (= M^T): the exact single carried state = base + tower.
    A copy (not a call) of _gdn2_dense_chunk's body -- the tested function must
    NOT be mutated (memory neuro_gdn2_dense_chunked)."""
    B, C, K = kn.shape
    work = kn.dtype
    dev = kn.device
    cs = torch.cumsum(LogA, dim=1)                                 # (B,C,K) <= 0
    Gt = torch.exp(cs)
    bk = b * kn
    if C % sub == 0:
        n_sub = C // sub
        csb = cs.view(B, n_sub, sub, K)
        d = (csb.unsqueeze(3) - csb.unsqueeze(2)).clamp(max=0.0)
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
                a_i = cs[:, lo - 1:lo]
                lam = torch.exp(cs[:, lo:hi] - a_i)
                rs = kn[:, :lo] * torch.exp(a_i - cs[:, :lo])
                A[:, lo:hi, :lo] = torch.bmm(bk[:, lo:hi] * lam, rs.transpose(1, 2))
                Bm[:, lo:hi, :lo] = torch.bmm(q[:, lo:hi] * lam, rs.transpose(1, 2))
    else:
        d = (cs.unsqueeze(2) - cs.unsqueeze(1)).clamp(max=0.0)
        Wp = torch.exp(d)
        A = torch.einsum('btsk,btk,bsk->bts', Wp, bk, kn)
        Bm = torch.einsum('btsk,btk,bsk->bts', Wp, q, kn)
    idx = torch.arange(C, device=dev)
    A = A * (idx.unsqueeze(1) > idx.unsqueeze(0)).to(work).unsqueeze(0)
    Bm = Bm * (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(work).unsqueeze(0)
    eye = torch.eye(C, device=dev, dtype=work).unsqueeze(0)
    RHS = v - torch.bmm(Gt * bk, S_full)                          # v is w*v pre-scaled
    U = torch.linalg.solve_triangular(A + eye, RHS, upper=False, unitriangular=True)
    r_in = torch.bmm(Bm, U)
    Gtq = Gt * q
    gC = cs[:, -1:]                                               # (B,1,K)
    k_carry = torch.exp(gC - cs) * kn                             # <= kn
    dS = torch.bmm(k_carry.transpose(1, 2), U)                   # (B,K,V), ref chunk end
    return U, r_in, Gtq, dS, gC


def gdn2_loglinear_chunked(v_seq, k_seq, q_seq, b_seq, w_seq, alpha_seq,
                           lw_seq, M_init, base_weight=None, chunk=64,
                           ckpt_chunks=False):
    """Chunkwise-parallel LOG-LINEAR GDN-2. Matches gdn2_loglinear_sequential
    (blk == chunk) to fp32 tolerance, and reduces to gdn2_chunked_dense when
    lw_seq == 1 and base_weight == 1 (the parity anchor).

    Carries BOTH the exact single state S_full (for the delta/erase solve --
    write path is bit-identical to gdn2_chunked_dense) AND a Fenwick tower of
    completed-chunk block-states {slot[l]} for the reweighted read. The whole
    tower is aged by Gamma_C each chunk on the SAME per-channel clock as S_full,
    so each block is bounded exactly like the proven single state -- no new
    unbounded exponentials (the design doc's overflow risk is tamed by this
    shared-reference scheme). Read at chunk ci:
        r = Bm@U  +  base_w*(Gtq@base) + sum_l lw^(l)*(Gtq@slot[l])
    with sum_l slot[l] + base == S_full so lw==base_w==1 telescopes to the
    single-state read. autocast hard-disabled (bf16 triangular inverse is
    unstable -- same rule as gdn2_chunked_dense).

    lw_seq: (B,T,L) per-level read weights; base_weight: (B,T) or None.
    Returns (r_out (B,T,d_v), S_final (B,d_v,d_k))."""
    from torch.utils.checkpoint import checkpoint as _ckpt
    B, T, D = v_seq.shape
    L = lw_seq.shape[-1]
    n_chunks = (T + chunk - 1) // chunk
    if 2 ** L <= n_chunks - 1:
        raise ValueError(f"L={L} levels too few for {n_chunks} chunks (need 2^L > n_chunks-1)")
    work = torch.float32
    ac_off = torch.autocast(device_type=v_seq.device.type, enabled=False)
    with ac_off:
        Q = q_seq.to(work)
        Bg = b_seq.to(work)
        WV = (w_seq * v_seq).to(work)                 # write gate folded into v
        Kr = k_seq.to(work)
        Kn = Kr / (Kr.norm(dim=-1, keepdim=True) + 1e-6)
        LogA = torch.log(alpha_seq.to(work).clamp(1e-6, 1.0))
        LW = lw_seq.to(work)
        BW = None if base_weight is None else base_weight.to(work)
        base = M_init.to(work).transpose(1, 2).contiguous()        # (B,K,V) base block
        S_full = base.clone()                                      # exact single state
        slot = [None] * L
        grad_on = ckpt_chunks and torch.is_grad_enabled() and any(
            t.requires_grad for t in (v_seq, k_seq, q_seq, b_seq, w_seq, alpha_seq))
        r_parts = []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            args = (S_full, Kn[:, start:end], WV[:, start:end], Q[:, start:end],
                    Bg[:, start:end], LogA[:, start:end])
            if grad_on:
                U, r_in, Gtq, dS, gC = _ckpt(_gdn2_ll_chunk, *args, use_reentrant=False)
            else:
                U, r_in, Gtq, dS, gC = _gdn2_ll_chunk(*args)

            # reweighted cross read: base (weight base_w) + tower levels (weight lw)
            lw_c = LW[:, start:end, :]                             # (B,C,L)
            bw_c = None if BW is None else BW[:, start:end].unsqueeze(-1)   # (B,C,1)
            cross = torch.bmm(Gtq, base)
            if bw_c is not None:
                cross = bw_c * cross
            for l in range(L):
                if slot[l] is not None:
                    cross = cross + lw_c[:, :, l:l + 1] * torch.bmm(Gtq, slot[l])
            r_parts.append(r_in + cross)

            # age the whole tower by Gamma_C (same clock as S_full), add dS, insert.
            gamma = torch.exp(gC).transpose(1, 2)                  # (B,K,1)
            base = base * gamma
            for l in range(L):
                if slot[l] is not None:
                    slot[l] = slot[l] * gamma
            S_full = S_full * gamma + dS
            carry = dS
            l = 0
            while carry is not None:
                if slot[l] is None:
                    slot[l] = carry
                    carry = None
                else:
                    carry = slot[l] + carry
                    slot[l] = None
                    l += 1
        r_out = torch.cat(r_parts, dim=1)
    return r_out.to(v_seq.dtype), S_full.transpose(1, 2).to(M_init.dtype)
