# Raising the Associative-Recall Ceiling — Design Doc

**Date:** 2026-09-03 (drafted overnight, autonomous)
**Status:** Research + ranked plan. NO code written (held for Glenn — the code
levers touch the shared trainer/kernel; not safe to build unattended while the
flagship runs). Config-only MQAR-curve test ran the same night (see §6).
**Motivation:** the head-to-head (`neuro_transformer_baseline_363m`) found ONE
real deficit vs the matched transformer: **high-N in-context recall** (spiking
single-pass COLLAPSES at N≥32; transformer degrades gracefully). This is the
only axis we clearly lose. This doc is how we attack it.

---

## 1. Diagnosis: it is an INTERFERENCE limit, not a state-size limit

New finding this session (decisive for what to build): at 30M the GDN-2 memory
already runs at **`n_heads=1`** — i.e. the **full dense `d_mem × d_mem` state**
(1024² ≈ 1M elements). A megabyte of state cannot literally be "too small" for
32 key→value bindings, yet single-pass recall still collapses at N≥32. So the
bottleneck is **cross-talk between overlapping key–value bindings** (retrieval
interference), not raw state dimension. This matches two prior nulls:
`neuro_500m_post_pretrain_refine_null` (4× budget didn't move it → not
retention/budget) and `neuro_decay_gate_saturation_null` (more forgetting
DESTROYS recall → not a decay problem). **Consequence:** the fix must reduce
interference / increase the expressivity of the write–read rule, OR extend
effective capacity at read time — NOT just enlarge the matrix.

Two properties any candidate must preserve (they are the project's moat):
- **Portable fixed-size state** (the ~17 KB saveable "brain"). A growing exact
  KV-cache (HOLA-style) breaks this — it's off-mission.
- **Constant cost in sequence length** (O(1)/token). Don't reintroduce O(n²).

---

## 2. Candidate levers, ranked by (evidence × on-target) ÷ (cost × risk)

| # | Lever | What it does | Cost / risk | Preserves moat? | Evidence |
|---|---|---|---|---|---|
| A | **MQAR curriculum** (range, not fixed-16) | Sets the trained-recall horizon to the capacity-matched sweet spot | **Free** (config) | ✅ | This session §6; arXiv:2507.02782 |
| B | **Split-replay as the default recall path** | 2-pass build-then-query; extends *effective* capacity at inference | **Free** (already built) | ✅ | `reference_split_replay` (2–4× at high N) |
| C | **DeltaProduct** (multi-step delta / token) | n_h gradient steps per token → diagonal+rank-n_h transition; more expressive write rule → less interference | **High** (chunked-kernel project) | ✅ (still fixed state) | arXiv:2502.10297 — better recall, state-tracking, **length-extrapolation**, and a **~20% faster** fwd; ref kernel at automl/DeltaProduct |
| D | **StateX post-training state expansion** | Expands a *pretrained* model's state with a short post-train | **Med** (post-train + code) | ✅ (bigger fixed state) | arXiv:2509.22630 — but gains modest for GLA/Mamba2 (+3.4%/+1.1%); may under-deliver for our *interference*-limited delta rule |
| E | **Mimetic initialization** | Init that makes the state mimic attention → learns copy/recall faster, ~2× length-gen | **Low** (init code), **fresh-pretrain only** | ✅ | arXiv:2410.11135 |
| F | Sparse Delta Memory / large sparse state | Sparse addressing to a big explicit memory | **High** | ⚠️ (drifts toward KV-cache) | arXiv:2607.07386 — **but SDM already CLOSED-NEGATIVE here** (`neuro_sdm_closed`); do not re-open without a strong new angle |

---

## 3. Why DeltaProduct is the standout architectural bet (lever C)

DeltaNet = one online-gradient step per token on the associative-recall loss.
Our GDN-2 is gated DeltaNet, so DeltaProduct is a **direct, composing extension**
of what we already run — not a foreign module. Taking n_h steps/token yields a
diagonal-plus-rank-n_h state transition (products of Householder reflections),
which is **strictly more expressive** at separating interfering bindings — i.e.
it attacks our exact bottleneck (§1). It keeps the state **fixed-size** (moat
intact), the recurrence is **stable** (every transition's spectral norm ≤ 1),
and the published chunked kernel is even **~20% faster** than DeltaNet's. It also
improves length-extrapolation, which is precisely the high-N generalization we
lack.

**Caveat / why gate it:** this is a **chunked-kernel project**, the same class as
log-linear — which just failed its gate after real effort. The project has ~11
module-lottery nulls. So DeltaProduct must go through the SAME discipline: a
kernel de-risking spike (parity vs a slow oracle + NaN watch + VRAM/throughput
at 30M) BEFORE any 90M/flagship commit, and a pre-registered recall win at
`--replay split` AND single-presentation N=32/64. The reference kernel
(automl/DeltaProduct) de-risks the hardest part.

---

## 4. Recommended sequence (cheapest, safest first — bank wins before big builds)

1. **NOW — free config/inference wins (no risk):**
   - **A:** set the flagship's MQAR to the capacity-matched *range* the §6 curve
     identifies (NOT fixed-16, NOT the naive [4,64] that hurt). Fold into
     flagship-v3.
   - **B:** make `--replay split` the standard recall-serving path and document
     it; it already buys 2–4× at high N for free. Consider a light replay-aware
     eval/report default.
2. **NEXT — cheap architectural de-risk (D):** try StateX-style post-training
   state expansion on the **existing flagship base** (`step_0124000`). It's the
   cheapest thing that could raise (not just relocate) the ceiling, and it runs
   on the trained model. **Gate:** recall up at N≥32 without hurting val_ppl or
   breaking the portable-state export. If the interference diagnosis (§1) is
   right, this may under-deliver — a cheap, informative test either way.
3. **THEN — the real bet (C): DeltaProduct**, gated behind a kernel spike as in
   §3. Highest upside; highest cost. Start with n_h=2.
4. **Fresh-pretrain only (E):** add mimetic init when flagship-v3 pretrains.

**Explicitly deferred:** F (SDM — closed-negative here); anything that grows an
exact KV-cache (breaks the moat); more "clever modules" outside this list.

---

## 5. Success criteria (pre-registered, per project convention)

For any lever: 2-seed 30M A/B, **single-presentation** recall at N=32/64/128 up
beyond the 2-seed spread (single-pres is where the in-context mechanism is
live), val_ppl within seed noise, throughput regression < 25%; then 90M confirm
→ flagship. Measure `--replay split` too, but the transformer-gap we're closing
is a single-pass gap, so single-pres is the primary bar.

## 6. Companion experiment run this session (config-only)

MQAR curriculum curve at 30M — `mqar_N_max ∈ {16, 32, 48, 64}`, one-variable off
the composed recipe, fine-grained recall. Answers lever A (the sweet spot) and
quantifies "the recall horizon tracks the trained range." Result table +
flagship recommendation: see the run's morning report / `neuro_mqar_range_30m_*`.

## 7. References
1. DeltaProduct — arXiv:2502.10297 (code: github.com/automl/DeltaProduct)
2. StateX post-training state expansion — arXiv:2509.22630 (code: github.com/thunlp/StateX)
3. Length-generalization in recurrent models (state coverage) — arXiv:2507.02782
4. Mimetic initialization for SSM recall — arXiv:2410.11135
5. Revisiting associative recall / LR-sensitivity — arXiv:2508.19029
6. Sparse Delta Memory (deferred; SDM closed here) — arXiv:2607.07386
7. Our diagnosis + prior nulls: `neuro_transformer_baseline_363m`,
   `neuro_decay_gate_saturation_null`, `neuro_500m_post_pretrain_refine_null`,
   `neuro_loglinear_stageA_kernel_2026_08_28` (kernel-gate discipline),
   `neuro_sdm_closed`, `reference_split_replay`.
