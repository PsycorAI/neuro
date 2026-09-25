# GDN-2 and OSDN — delta-rule recall upgrades (design)

Date: 2026-07-11. Status: building. Both are 30M screens gated behind default-off
flags, verified no-op at flag-off before any GPU time, sequential-loop kernel
first (chunked WY kernel deferred until a screen wins).

Motivation: the 500M post-pretrain refines (MQAR/sparsity) came back null on
recall even at 4× budget — evidence the delta-rule memory *core* is the binding
constraint on low-N recall, not the refine budget. Both papers below attack that
core directly, near our scale.

---

## Notation / layout mapping

Our memory `M` is shape `(B, d_v, d_k)` with `d_v == d_k == d_mem`, read as
`r = M @ q`, written `M += u ⊗ k`. The papers write their state `S` in the
**[key, value]** layout `(d_k, d_v)`, so **`M = Sᵀ`**. All equations below are
re-derived into our `M` layout. Keys are L2-normalized (`kn = k/(‖k‖+1e-6)`), as
in the existing `delta_rule` path.

---

## GDN-2 (Gated DeltaNet-2, arXiv:2605.22791, NVLabs)

Decouples the single scalar that standard gated delta uses for two different
jobs — **erasing** old content (key side) and **writing** new content (value
side) — into independent channel-wise gates, plus channel-wise decay.

Paper state update (their [key,value] layout), Eq. 10:

    S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t-1} + k_t (w_t ⊙ v_t)ᵀ
    D_t = diag(α_t)                         # channel-wise decay on the key dim

Gates (elementwise sigmoids of projections of the input `x_t`; here `x_t = spk`):

    b_t = σ(W_b x_t)          ∈ (0,1)^{d_k}      erase gate
    w_t = σ(W_w x_t)          ∈ (0,1)^{d_v}      write gate
    g_t = −exp(a) ⊙ softplus(W_f x_t + δ) ;  α_t = exp(g_t) ∈ (0,1)^{d_k}   decay

Reduces to KDA when `b_t=w_t` tie to a scalar, and to Gated DeltaNet when the
decay also collapses to a scalar.

### Re-derived into our `M = Sᵀ` layout (per-timestep)

    kn      = k / (‖k‖ + 1e-6)                      # (B, d_k)
    α_t     = exp(−exp(a) * softplus(W_f(spk)))     # (B, d_k), a: Param(d_mem)
    b_t     = σ(W_b(spk))                            # (B, d_k)
    w_t     = w_floor + (1−w_floor)·σ(W_w(spk))     # (B, d_v)  floored (see below)
    M_dec   = M * α_t[:, None, :]                    # scale key-columns (decay)
    r_erase = M_dec @ (b_t ⊙ kn)                     # (B, d_v)  read at gated key
    z       = w_t ⊙ v                                # (B, d_v)  write target
    u       = z − r_erase                            # (B, d_v)
    M       = M_dec + u ⊗ kn                          # write at plain key
    r       = M @ q                                   # read

Verified by hand that with `b_t=w_t=β` (scalar) and `α_t=α` (scalar) this is
exactly our current gated `delta_rule` step (`u = β(v − αM·kn)`), so the
sequential reference has a clean reduce-to-baseline check.

**β-floor carried forward.** The hard-won lesson (`beta_floor: 0.1`, scalar-β
write-gate collapse) ports directly: apply the floor to the *write* gate `w_t`
so it cannot saturate to zero. Reuse the existing `beta_floor` config value as
`w_floor`. The erase gate `b_t` is left unfloored (b→0 just disables the delta
correction, degrading to gated-Hebbian — not a silent-zero-memory pathology).

**Params added vs current gated delta:** `W_b, W_w, W_f` each `Linear(N, d_mem)`
and `a: Param(d_mem)`, replacing the scalar `W_beta`/`W_alpha`. Modest; note in
the A/B config (the win must be the mechanism, not the parameters — a KDA-only
follow-up arm can isolate decoupling-vs-channel-decay if GDN-2 clears).

---

## OSDN (Online preconditioned delta rule, arXiv:2605.13473)

Adds a **diagonal preconditioner** `d_t ∈ R^{d_k}` to the delta write: the error
is read with the plain key, but the rank-1 write is placed along the
preconditioned key `k̃_t = d_t ⊙ k_t`. The preconditioner is **not a learned
parameter** — it is adapted online per step by a closed-form hypergradient, so
it adds ~zero params and only a cheap per-step vector state. Reported +32%
in-context recall at 340M.

Paper update (their layout):

    S_t = S_{t-1}(I − β_t k_t k̃_tᵀ) + β_t v_t k̃_tᵀ ,   k̃_t = d_t ⊙ k_t

which expands to `S_t = S_{t-1} + β_t (v_t − S_{t-1} k_t) k̃_tᵀ` — read/erase on
plain `k_t`, write outer-product on preconditioned `k̃_t`.

Online preconditioner (elementwise on the key dim; `k² = k⊙k`):

    d_{t+1} = d_t + (η β_t / max(‖k_t‖², ε)) · (1 − β_t ⟨d_t, k²⟩) · k²
    d_{t+1} = clamp(d_{t+1}, d_min, d_max)          # box [0.5, 2.0]

Defaults from the paper: `η = 0.003`, box `[0.5, 2.0]`, small `ε`. Init `d_0 = 1`
(identity preconditioner ⇒ exact standard delta rule at t=0).

### Re-derived into our `M = Sᵀ` layout (per-timestep)

    kn      = k / (‖k‖ + 1e-6)                      # (B, d_k); note ‖kn‖²≈1
    β       = beta_floor + (1−beta_floor)·σ(W_beta(spk))   # (B,1), existing
    prevM   = α · M                                  # existing decay (decay_gate/lam)
    Mk      = prevM @ kn                             # read at plain key
    u       = β · (v − Mk)                           # (B, d_v)
    k̃      = d_t ⊙ kn                               # preconditioned write key
    M       = prevM + u ⊗ k̃
    r       = M @ q
    # online preconditioner (carried across the T-loop within a window):
    k2      = kn ⊙ kn                                # (B, d_k)
    nrm     = clamp(Σ k2, ε)                         # ‖k‖² (≈1 for normalized)
    dot     = Σ (d_t ⊙ k2)                           # ⟨d_t, k²⟩
    d_t     = clamp(d_t + (η β / nrm)(1 − β·dot)·k2, d_min, d_max)

**Orthogonal to gating** — composes with the existing `decay_gate` + `beta_floor`
recipe; it only changes the write-side key. **Params added: none.** `d_t` is a
state, reset to 1 at the start of each `forward_sequence` window (a minor
approximation vs carrying it across truncated-BPTT windows; fine for a screen,
and `d` re-adapts within a few tokens). Because `d_t` depends only on the key
sequence, a future chunked kernel can precompute it in one pass, then feed
`k̃ = d ⊙ k` as the write-side key into an asymmetric-key WY solve.

---

## Implementation plan (both)

1. Sequential memory loop inside `SpikingHebbianBlock.forward_sequence` (LIF and
   all projections stay FPT-parallel; only the O(T) memory update loops) — the
   exact structure the existing `vector_beta` branch already uses.
2. `step()` (token-by-token generation path) raises `NotImplementedError` for
   these two modes for now — fail loud rather than run the wrong math; add the
   step + chunked kernels only if a screen wins.
3. Reference O(T) loops `gdn2_sequential` / `osdn_sequential` added to
   `delta_chunked.py` as test oracles, with a reduce-to-baseline unit check
   (GDN-2 with tied scalar gates == `delta_sequential_gated`; OSDN with `d≡1` ==
   `delta_sequential`).
4. Config flags (all default-off, so flag-off adds zero state_dict keys and is
   byte-identical to the current path):
   - `gdn2_enabled: false`
   - `osdn_enabled: false`, `osdn_eta: 0.003`, `osdn_dmin: 0.5`, `osdn_dmax: 2.0`
5. Guards: both require `delta_rule: true`; both incompatible with `n_heads>1`,
   `vector_beta`, `titans`; GDN-2 and OSDN mutually exclusive.

## Test / gate

30M A/B, current gated-delta recipe (delta_rule + decay_gate + beta_floor:0.1,
MQAR-aug on) as control. Pass = val_ppl win > ~1.5 ppl noise floor **and** a
memory-Δ / recall_eval lift, then a seed1 confirm before either folds into the
next 500M. Screen GDN-2 and OSDN in parallel (5080 + a freed slot); if both
clear, they compose (OSDN's preconditioner over GDN-2's decoupled write) — test
the combination at 90M before the 500M bake.
