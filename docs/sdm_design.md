# Sparse Delta Memory (SDM) — design doc

Date: 2026-07-12. Status: design (no code yet). Paper:
[arXiv:2607.07386](https://arxiv.org/abs/2607.07386), Meta FAIR, Jul 8 2026.

Now the top strategic priority in the queue: Test 2's null (2026-07-12) settled
that the low-N recall ceiling of our current delta-rule memory is architectural,
not budget. GDN-2 and OSDN improve the *write rule* but leave state capacity
fixed at `d_v × d_k`. SDM is the only backlog item that attacks capacity itself,
scaling the recurrent state orders of magnitude under an isoFLOP / iso-param
constraint. This doc scopes the build; NO code yet — verify-before-run
discipline (fracconn-style identity-init check first).

---

## The mechanism (mapped from the paper into our layout)

Standard delta rule (ours today, per layer):

    M ∈ R^(d_v × d_k)                 # single dense outer-product state
    M_t = α_t · M_{t-1} + β_t · (v_t − α_t M_{t-1} k̂_t) k̂_tᵀ    # dense write
    r_t = M_t q_t                       # dense read

SDM replaces the dense `M` with a **memory bank of N slots**, and each timestep
touches only the top-W slots for writing and top-R slots for reading:

    M ∈ R^(N × d)                       # N slots, each a d-dim value vector
    (learned initial state M_0, trainable parameter; the "long-term store")

**Slot addressing (Product-Key Memory, arXiv:1907.05242):**

    k'_t = W_k x_t ∈ R^(2√N)            # write-side query, split into 2 halves
    q'_t = W_q x_t ∈ R^(2√N)            # read-side query, split into 2 halves
    write_scores = k'_1,t ⊕ k'_2,t ∈ R^(√N × √N)   # outer SUM (not product)
    read_scores  = q'_1,t ⊕ q'_2,t ∈ R^(√N × √N)
    I^w_t = top-W(write_scores)         # W indices ∈ [1..N]
    I^r_t = top-R(read_scores)          # R indices ∈ [1..N]

The outer-*sum* structure lets top-K over N slots be done in O(√N) time instead
of O(N) — the trick that makes N=10^6+ slots practical.

**Per-slot gated delta update (only for the W selected write slots):**

    M̃_t[i] = α_t · M_{t-1}[i]                        # channel-wise decay
    M_t[i] = M̃_t[i] + β_t · k_t^(i) · (v_t − M̃_t[i])   # gated delta write
              # applied ONLY for i ∈ I^w_t; unselected slots unchanged
    α_t = exp(−A · softplus(W_α x_t + b_dt))          # channel-wise forget
    β_t = σ(W_β x_t)                                   # write gate (scalar per slot)
    k_t^(i) = per-slot write coefficient (from the same PKM scores)

**Read (only from the R selected read slots):**

    y_t = Σ_{i ∈ I^r_t} q_t^(i) · M_t[i]

Everything else — LIF front-end, projections, layer stacking, FF residual,
tokenizer, KD, MTP heads, Engram v2 — is untouched. SDM slots in as a
replacement `SpikingSDMBlock` alongside the existing `SpikingHebbianBlock`,
selected by config flag (mutually exclusive with `delta_rule` / `gdn2_enabled` /
`osdn_enabled` — SDM IS a gated-delta variant).

---

## Composes with existing wins

| Existing feature       | Composes with SDM? | Notes |
|------------------------|--------------------|-------|
| `beta_floor: 0.1`      | Yes — direct port  | Apply floor to per-slot β exactly as today (still a scalar-sigmoid, same collapse pathology risk). |
| `decay_gate`           | Yes — direct port  | Channel-wise α is already SDM's decay recipe; matches paper. |
| NorMuon                | Yes — orthogonal   | Optimizer-level; touches all matrices including new `W_k`, `W_q`, `M_0`. |
| Engram v2              | Yes — orthogonal   | Engram v2 injects into residual stream at a different point. |
| Frac-connections       | N/A — closed neg   | (Not baking regardless.) |
| MoR router / recurrent_depth | Yes — orthogonal | SDM replaces the memory rule inside a block; the router still gates whole-block application. |
| MTP heads              | Yes — orthogonal   | Auxiliary loss on final logits, untouched. |
| GDN-2 / OSDN           | Composable as write-rule variants of SDM's per-slot update — NOT mutually exclusive | Amended 2026-07-12. The prior "same-M variant" framing over-generalizes: SDM's per-slot update is a rank-1 gated delta write in exactly the same shape GDN-2 decouples (erase/write) and OSDN preconditions. Both port unchanged into the top-W selected slots. If GDN-2 clears seed1 (mid-run signals suggest it will), spec a phase-6 variant `sdm_write_rule: gdn2 \| osdn \| gated_delta` rather than screen them as competitors. The head-to-head is between *addressing schemes* (dense vs sparse-slot), not between write rules. |

---

## Sizing (30M screen → 500M restart)

The paper's 8B ref uses N ≈ 4096 slots × d = ??? per slot, giving ~157M memory
params per layer. For a **30M screen** where our whole model is ~44M, we want
memory params ~2–5M so the state comparison is meaningful but not swamping the
model:

- **N = 1024, per-slot d = 1024, W = 4, R = 4**: 1M memory params, √N = 32 →
  PKM outer-sum is 32×32 = 1024 scores. Top-4 selection ~O(√N + W log W).
- **Measured (phase 3, 2026-07-12), not estimated:** `configs/screen_sdm_ab_treatment_30m.yaml`
  builds at **46.43M params vs the shared control's 44.07M — +2.36M added**,
  notably less than this doc's original ~6M estimate. The gap: phase 3's
  actual design reuses the block's existing `W_beta`/`W_alpha` (already
  counted in the 44.07M base) rather than giving SDM its own gate matrices
  as this section originally assumed. Real breakdown per layer: `M_0`
  (N×d = 1024×1024 ≈ 1.05M) + `pkm_w.W_pkm` + `pkm_r.W_pkm` (2 × 1024×64 ≈
  131K) ≈ 1.18M/layer × 2 layers ≈ 2.36M. Same order as GDN-2's addition
  (50.37M − 44.07M ≈ 6.3M) but smaller, since GDN-2 adds 3 full `Linear(N,d_mem)`
  gate matrices per layer where SDM adds none.

For a **500M restart**, scale to N=16384, per-slot d=2048, W=8, R=8 → ~33M
memory params × 5 layers = ~165M memory, ~35% of the model's non-embedding
budget. Match the paper's scaling curve as a sanity check before commit.

**Sizing footnote (added 2026-07-12).** The 500M "architectural recall ceiling"
verdict from Test 2 (see journey §37) was measured on a model whose *effective*
per-layer memory state is smaller than a naive `d_v × d_k = 2048²` count
suggests. `configs/demo_phase1b_500m_v2_fast.yaml` has `use_fla_kernel: true`,
and `delta_fla_gated` (in `src/delta_chunked.py`) splits d_mem=2048 into H=8
diagonal blocks of 256×256 (fla's head-dim cap): effective state ≈ 8·256²=524K
params/layer, ~8× less than the dense 4.19M/layer that our 30M eager screens run.
This *strengthens* the SDM thesis (the ceiling is even more of a state-capacity
constraint than an ambient reading of it implies) — but it also means the SDM
sizing table above should compare the 165M SDM state against the *effective*
memory state of whatever kernel the 500M restart uses. If the restart also
uses fla, the "35% of non-embedding" comparison should count SDM's real state
vs the block-diagonal effective state, not against the notional dense state
the code appears to allocate. Decision: settle "fla-block-diagonal vs dense" as
an explicit sub-question of the 500M restart plan before committing (dense
adds VRAM and precludes fla; SDM's sparse addressing sidesteps the head-dim
cap entirely and may be the better route to full-rank state at this scale).

---

## Build phases

1. **Design + literature check (this doc).** Confirm PKM addressing details vs
   the original PKM paper (arXiv:1907.05242) so the outer-sum score
   construction is exact, not paraphrased. ← DONE.

2. **`src/sdm.py`** — LANDED 2026-07-12. `PKMIndex` module (outer-sum score
   construction with per-sub top-M candidate + global top-K, O(K·√N)) and
   `SparseSDMMemory` module (owns M_0 zero-init + separate write/read PKM
   heads; reuses the block's existing β / α / v projections so beta_floor
   telemetry and decay_gate keep working unchanged). Fracconn-style CPU
   verification (`temp/test_sdm.py`, 22/22 checks pass): PKM outer-sum
   invariant + top-K correctness vs dense oracle at K∈{1,4,8}; SDM zero-init
   + β=0 exact-zero output; β=0 closed-form deterministic read; α=0
   fast-forget correctness; grad flows through M_0 and both PKM heads;
   determinism; autocast bf16 finiteness; slot-selection telemetry populated.
   **Correction to the phase-2 language above:** the "dense_all_slots reduces
   to gated delta" claim was aspirational and does NOT hold — SDM's per-slot
   (N, d) memory is structurally different from delta's (d_v, d_k) shared
   outer-product state. The identity checks landed instead prove correctness
   without a reduce-to-baseline oracle, matching the fracconn discipline in
   spirit.

3. **`src/model.py` integration — LANDED 2026-07-12.** `sdm_enabled: false` +
   sizing knobs (`sdm_N`, `sdm_W`, `sdm_R`) added to `SpikingHebbianBlock` and
   `SpikingHebbianLM`; config plumbing added to `train_gpu.build_model` and
   `recall_eval.build`. Guards: requires `delta_rule: true`; mutually
   exclusive with `gdn2_enabled`/`osdn_enabled`/`vector_beta`/`titans`/
   `n_heads>1`. Reuses the block's existing `W_beta`/`W_alpha` (β/α stay
   block-owned so `beta_floor` telemetry keeps working, per the composition
   table). `step()` raises `NotImplementedError` (no token-by-token path yet,
   same as GDN-2/OSDN). **Simplification vs the original phase-3 sketch:**
   rather than a per-block `init_state()` refactor of the LM-level state
   allocator, phase 3 constrains `sdm_N == d_mem` (enforced, raises otherwise)
   so the state stays square (N, N) and the existing `(d_state, d_state)`
   square-state allocator works completely unchanged — zero changes to
   `_fpt_stack_pass` / `forward()`. This exactly matches the 30M screen's
   already-scoped sizing (N=1024=d_mem). The 500M restart's non-square sizing
   (N=16384, d_mem=2048) still needs the state-allocator refactor as a
   separate future phase — not required to run the 30M screen.
   **Verified:** `temp/test_sdm_wiring.py`, 21/21 checks — flag-off bit-
   identical state_dict + forward vs baseline; flag-on 1-layer and 2-layer
   builds, forward+backward finite, grad flows into both PKM heads (note:
   `SparseSDMMemory.M_0` is correctly gradient-free in this wired path — the
   LM always supplies `M_init` externally, so `M_0` only activates when the
   module is driven standalone, as phase 2's CPU tests do); all 7 guards
   fire correctly; `sdm_N` defaults to `d_mem` when omitted.
   `configs/screen_sdm_ab_treatment_30m.yaml` built (byte-identical to the
   shared GDN-2/OSDN control except `sdm_*` flags) and verified end-to-end
   through both `train_gpu.build_model` and `recall_eval.build` — 46.43M
   params vs the control's 44.07M (+2.36M, see sizing section above).

4. **`src/train_gpu.py` + `scripts/recall_eval.py`** — config plumbing
   back-port (same pattern as GDN-2/OSDN did).

5. **Verify at 30M-config CPU scale** — same `temp/test_sdm.py` battery:
   dense-oracle reduction, flag-off state_dict identical, flag-off forward
   bit-identical, flag-on trains with finite grads, guards fire, grad flows
   through the top-k selection (index-scatter grad reaches `W_k`).

6. **30M screen** — `screen_sdm_ab_treatment_30m.yaml`, byte-identical to the
   GDN-2/OSDN control except `sdm_enabled: true` and the SDM sizing knobs.
   Gate: val_ppl win > ~1.5 ppl noise floor *and* memory-Δ / recall_eval lift
   at N=4/8/16 (the range the 500M's null Test 2 couldn't move — the whole
   point). Then seed1 confirm.

7. **90M scale-up** if 30M clears seed1. This is where SDM should start to
   pull away, per the paper's isoFLOP curves.

8. **500M restart bake** — only if 90M also clears. SDM sizing at 500M
   ~165M memory-state params (~35% of the model non-embedding budget), close
   to the paper's 8B ratio.

---

## Risk register

- **Sparse-scatter gradient correctness.** Torch's `index_add_` has correct
  gradients but not always in the vectorized form needed for autograd tape.
  Verify with a finite-difference check on `M_0` grads in phase 5.
- **Dead slots.** If a slot is never selected during training, its gradient
  is zero and it stays at init forever. Paper reports this is *fine*
  (unused capacity, not a bug), but our telemetry should count per-slot
  selection frequency so we notice degenerate "1 slot does everything"
  collapse (analogous to router collapse in MoE).
- **Grad-checkpointing.** Only-top-W-slots-touched-per-step means the tape
  is naturally sparse; the ~50GB blowup we hit with GDN-2 shouldn't recur.
  But confirm empirically at real scale before committing to a big screen.
- **PKM top-k determinism under autocast.** Ties in scores can flip which
  slot wins under bf16 vs fp32. Cache scores in fp32 for the top-k, do
  the write in the training dtype.
- **Interaction with recurrent_depth.** The M state is carried across the
  block-loop passes today. SDM's per-slot state has the same shape as the
  paper's, so this should be structural — but verify the per-pass memory
  layout in phase 3.

---

## Related work briefly considered and NOT chosen

- **Fast-weight Product Key Memory** (arXiv:2601.00671) — the direct
  ancestor of SDM but without the gated-delta write rule. SDM strictly
  subsumes it in our stack.
- **Titans** (arXiv:2501.00663) — sequential memory with per-timestep
  gradient-descent updates; we already have this behind the `titans` flag
  and it screened flat.
- **StateX** (arXiv:2509.22630) — post-training state expansion, similar
  goal but as a refine step; MQAR-refine's Test 2 null suggests refine-only
  state expansion won't move recall for our architecture. Different lever
  entirely, but banks a fallback if SDM is too invasive.
