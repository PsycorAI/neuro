"""Sparse Delta Memory (SDM) — arXiv:2607.07386 (Cabannes et al., Meta FAIR,
2026-07-08). Standalone module; model.py wiring is a separate change gated
behind `sdm_enabled: false`.

Replaces the dense (d_v × d_k) gated-delta memory with an N-slot bank
M ∈ R^(N × d), addressed by Product-Key Memory (arXiv:1907.05242): outer-SUM
of two √N-way sub-scores, top-K in O(√N) instead of O(N). Per-slot gated
delta write on the top-W selected slots; read from the top-R selected slots.

The recurrent state grows orders of magnitude under an isoFLOP / iso-param
constraint (N slots × d per slot vs d_v × d_k dense) — the only backlog item
that attacks state *capacity* rather than the delta write rule.
See docs/sdm_design.md.

Note on the "dense_all_slots reduces to gated delta rule" phrasing in the
design doc: that reduction was aspirational and does NOT hold structurally.
Standard delta rule's M ∈ R^(d_v × d_k) is a single dense outer-product state
that all timesteps share; SDM's M ∈ R^(N × d) is N independent d-dim value
vectors with per-slot updates. These are structurally different memories at
any parameter setting. What's true is only that both are linear-RNN memories
addressed by learned queries — not that one is a special case of the other.
The identity checks below are the fracconn-style pre-GPU verification for
this module: correctness of the addressing mechanism, safe zero-init no-op,
gradient flow into all parameters, deterministic top-k, and autocast
robustness. They are NOT a reduce-to-delta oracle.

Composition (see docs/sdm_design.md):
  - beta_floor: direct port (β stays scalar per timestep, sigmoid on W_β)
  - decay_gate: direct port (α stays scalar per timestep, applied to all slots)
  - NorMuon: orthogonal (touches this module's Linear layers too)
  - Engram v2: orthogonal (different injection point)
  - GDN-2 / OSDN: composable as per-slot write-rule variants of the update
    inside — NOT mutually exclusive as originally framed. See composition
    matrix update in docs/sdm_design.md.
"""
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# SDM grad-checkpointing chunk size. Same pattern as _GDN2_OSDN_CHUNK in
# model.py: the per-timestep loop persists (B, N, d) M state across ~T
# timesteps for BPTT. At B=32, N=1024, d=1024, T=512 the naive tape hits
# ~65 GB even before block-stacking. Checkpointing per chunk of 32 cuts peak
# activation memory ~T/32 = 16x for a ~2x compute hit (one recompute per
# chunk during backward). Discovered live on 2026-07-13 when the first SDM
# 30M screen OOM'd at step 1 with 45 GB allocated -- same class of pathology
# that blocked GDN-2/OSDN before their grad-ckpt fix.
_SDM_CHUNK = 32


def _sdm_chunk_fwd(M, v_ch, beta_ch, alpha_ch,
                    w_scores_ch, w_idx_ch, r_scores_ch, r_idx_ch):
    """Per-chunk sequential SDM loop. Pure function of its tensor inputs --
    safe target for torch.utils.checkpoint.checkpoint(use_reentrant=False).
    Returns (M_final, r_chunk). Byte-identical math to the pre-refactor
    forward's inner loop; only the tape structure differs (recompute per
    chunk instead of one giant tape across all T timesteps)."""
    r_list = []
    d = v_ch.size(-1)
    C = w_idx_ch.size(1)
    for t in range(C):
        a = alpha_ch[:, t, :].unsqueeze(-1)            # (B,1,1)
        M = a * M                                       # decay all slots
        b_t = beta_ch[:, t, :]                          # (B,1)
        wi = w_idx_ch[:, t]                             # (B,W)
        ws = w_scores_ch[:, t]                          # (B,W)
        wi_e = wi.unsqueeze(-1).expand(-1, -1, d)       # (B,W,d)
        M_at_w = M.gather(1, wi_e)                      # (B,W,d)
        v_t = v_ch[:, t, :].unsqueeze(1)                # (B,1,d)
        diff = v_t - M_at_w                              # (B,W,d)
        u = (b_t.unsqueeze(-1) * ws.unsqueeze(-1)) * diff  # (B,W,d)
        M = M.scatter_add(1, wi_e, u)                   # top-W gated delta write
        ri = r_idx_ch[:, t]                             # (B,R)
        rs = r_scores_ch[:, t]                          # (B,R)
        ri_e = ri.unsqueeze(-1).expand(-1, -1, d)       # (B,R,d)
        M_at_r = M.gather(1, ri_e)                      # (B,R,d)
        r_t = (rs.unsqueeze(-1) * M_at_r).sum(dim=1)    # (B,d)
        r_list.append(r_t)
    return M, torch.stack(r_list, dim=1)


class PKMIndex(nn.Module):
    """Product-Key Memory addressing over N slots (arXiv:1907.05242).

    Given input x, produces the top-K (score, slot_index) pairs using the
    outer-SUM trick — top-K over N in O(K·√N + K²·log K), not O(N):

        sub = W_pkm(x) ∈ R^{2√N}
        s1 = sub[..., :√N]   (first  sub-score, √N-way)
        s2 = sub[..., √N:]   (second sub-score, √N-way)
        score[i·√N + j] = s1[i] + s2[j]     (row-major flat slot index)

    Then take the per-sub top-M (M = K · mult) and combine into M² candidates
    (M ≤ √N by construction), and take global top-K from those. mult=4 is
    enough to catch the true top-K under any realistic score distribution
    (the sole failure mode is a score at (i,j) where BOTH i∉top-M(s1) AND
    j∉top-M(s2), which is exponentially unlikely for K = 4-8 and mult ≥ 4).

    Args:
        n_neurons: input dim (spikes / hidden)
        N: number of slots, must be a perfect square
        mult: candidate-width multiplier over K (default 4)
    """

    def __init__(self, n_neurons, N, mult=4):
        super().__init__()
        sqrtN = int(math.isqrt(N))
        assert sqrtN * sqrtN == N, f"N={N} must be a perfect square"
        self.N = N
        self.sqrtN = sqrtN
        self.mult = mult
        # Two sub-halves, one linear projection produces both — no bias so the
        # unbiased zero-input case gives zero scores (helps the deterministic
        # tie-break story stay clean).
        self.W_pkm = nn.Linear(n_neurons, 2 * sqrtN, bias=False)

    def forward(self, x, K):
        """x: (..., n_neurons); K: int (top-K to return).
        Returns (scores, slot_idx), each shape (..., K).
        slot_idx values ∈ [0, N).
        """
        s = self.W_pkm(x)                                 # (..., 2√N)
        s1 = s[..., :self.sqrtN]
        s2 = s[..., self.sqrtN:]
        M = min(self.sqrtN, K * self.mult)
        v1, i1 = s1.topk(M, dim=-1)                       # (..., M)
        v2, i2 = s2.topk(M, dim=-1)
        # Outer sum of the per-sub top-M candidates: (..., M, M)
        cand = v1.unsqueeze(-1) + v2.unsqueeze(-2)
        cand_flat = cand.reshape(*cand.shape[:-2], M * M)
        top_vals, top_flat = cand_flat.topk(K, dim=-1)    # (..., K)
        top_i_m = top_flat // M
        top_j_m = top_flat % M
        top_i = i1.gather(-1, top_i_m)                    # (..., K) in [0,√N)
        top_j = i2.gather(-1, top_j_m)
        slot_idx = top_i * self.sqrtN + top_j              # (..., K) in [0,N)
        return top_vals, slot_idx


class SparseSDMMemory(nn.Module):
    """N-slot sparse gated delta memory.

    Owns:
      - `M_0`: (N, d) learned initial slot state, zero-init for safe residual
      - `pkm_w`, `pkm_r`: separate write-side and read-side PKM heads

    Does NOT own β / α — those stay on the containing block so the
    existing `beta_floor` telemetry and `decay_gate` config switch keep
    working unchanged. Caller passes `beta_seq` and `alpha_seq` in.

    Args:
        n_neurons: input dim for addressing
        N: number of memory slots (perfect square)
        d: per-slot value dim (typically d_mem)
        W: top-W slots written per timestep
        R: top-R slots read per timestep

    forward(x_seq, v_seq, beta_seq, alpha_seq, M_init=None):
        x_seq:     (B, T, n_neurons)   spikes for addressing
        v_seq:     (B, T, d)           value stream to write
        beta_seq:  (B, T, 1)           scalar write gate per timestep (post-floor)
        alpha_seq: (B, T, 1)           scalar decay gate per timestep, in (0,1)
        M_init:    (B, N, d) or None   carried state; None => broadcast M_0
        returns:   r_seq (B, T, d), M_final (B, N, d)
    """

    def __init__(self, n_neurons, N, d, W=4, R=4):
        super().__init__()
        self.N = N
        self.d = d
        self.W = W
        self.R = R
        self.pkm_w = PKMIndex(n_neurons, N)
        self.pkm_r = PKMIndex(n_neurons, N)
        # Zero-init M_0: all initial slot reads are zero-vectors → SDM
        # contributes nothing to the residual at step 0 (safe drop-in). The
        # model learns the initial slot content during training.
        self.M_0 = nn.Parameter(torch.zeros(N, d))
        # Per-slot selection counter (non-persistent, telemetry only). Detects
        # dead-slot collapse (analog of MoE router collapse in the risk
        # register). Not part of any loss.
        self.register_buffer(
            "_slot_select_count",
            torch.zeros(N, dtype=torch.long),
            persistent=False,
        )

    def _init_M(self, B, device, dtype):
        return self.M_0.unsqueeze(0).expand(B, self.N, self.d).to(dtype).contiguous()

    def forward(self, x_seq, v_seq, beta_seq, alpha_seq, M_init=None):
        B, T = x_seq.shape[:2]
        device = x_seq.device
        dtype = v_seq.dtype
        M = self._init_M(B, device, dtype) if M_init is None else M_init

        # PKM addressing precomputed for all t (no state dependence on M).
        # Cast to fp32 for the top-k so bf16 ties don't flip which slot wins
        # across otherwise-identical runs; scores then cast back to the training
        # dtype for the write/read math. Notes in risk register.
        with torch.autocast(device_type=x_seq.device.type, enabled=False):
            x_fp = x_seq.float()
            w_scores, w_idx = self.pkm_w(x_fp, K=self.W)   # (B,T,W), (B,T,W)
            r_scores, r_idx = self.pkm_r(x_fp, K=self.R)   # (B,T,R), (B,T,R)
            # Softmax-normalize top-K scores (standard PKM; arXiv:1907.05242).
            # Raw logits are unbounded -- without this, write/read weights can
            # grow without limit and blow up M over training (observed: NaN
            # loss at step ~400 in screen_sdm_ab_treatment_seed1_30m).
            w_scores = w_scores.softmax(dim=-1)
            r_scores = r_scores.softmax(dim=-1)
        w_scores = w_scores.to(dtype)
        r_scores = r_scores.to(dtype)

        # Telemetry: per-slot selection frequency. Detached; diagnostic only.
        if self.training:
            with torch.no_grad():
                flat_w = w_idx.reshape(-1)
                self._slot_select_count.index_add_(
                    0, flat_w, torch.ones_like(flat_w))

        # Chunked per-timestep loop with grad-checkpointing during training.
        # Each chunk runs _sdm_chunk_fwd (module-level pure function) either
        # directly (eval mode) or under checkpoint (train mode). Peak
        # activation memory drops T/_SDM_CHUNK ~= 16x; backward recompute
        # cost ~2x. Bit-exact vs a single-chunk forward (verified in
        # temp/test_sdm.py).
        r_chunks = []
        for s in range(0, T, _SDM_CHUNK):
            e = min(s + _SDM_CHUNK, T)
            args = (M, v_seq[:, s:e], beta_seq[:, s:e], alpha_seq[:, s:e],
                    w_scores[:, s:e], w_idx[:, s:e],
                    r_scores[:, s:e], r_idx[:, s:e])
            if self.training:
                M, r_ch = checkpoint(_sdm_chunk_fwd, *args, use_reentrant=False)
            else:
                M, r_ch = _sdm_chunk_fwd(*args)
            r_chunks.append(r_ch)
        r_out = torch.cat(r_chunks, dim=1)
        return r_out, M

    @torch.no_grad()
    def slot_stats(self):
        """Per-slot selection frequency snapshot (write-side). Returns
        dead-slot fraction, coverage (frac of slots ever selected), and
        selection Gini (0 = uniform, 1 = one slot wins everything)."""
        c = self._slot_select_count.float()
        total = c.sum().item()
        if total == 0.0:
            return {"dead_frac": 1.0, "coverage": 0.0, "gini": 0.0}
        p = c / total
        dead_frac = float((c == 0).float().mean().item())
        coverage = 1.0 - dead_frac
        # Gini = 1 − Σ p²  is the Simpson complement, well-defined for 0-mass
        # slots and cheap. 0 = perfectly uniform; approaches 1 as one slot
        # dominates.
        gini = float(1.0 - (p * p).sum().item())
        return {"dead_frac": dead_frac, "coverage": coverage, "gini": gini}
