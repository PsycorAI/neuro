# Log-Linear Attention on GDN-2 — Design Doc

**Date:** 2026-08-27
**Status:** Scoped, NOT built. Recommendation: **de-risk with a kernel spike BEFORE any flagship commit; do flagship-v2 (Tier-1) first regardless.**
**Related:** `docs/improvement_plan_2026_08.md` (Tier 2), memory `neuro_gdn2_dense_chunked`, `neuro_decay_gate_saturation_null`, `neuro_500m_post_pretrain_refine_null`

---

## 1. What it is

Replace GDN-2's single fixed-size memory matrix `S` (B,K,V) with a **hierarchy of
O(log T) states at geometric temporal resolutions** (Fenwick/segment-tree layout),
read out with data-dependent per-level weights. This lifts the fixed-state
associative-recall ceiling — the ONE weakness this project has proven is
architectural (4x budget didn't move it: `neuro_500m_post_pretrain_refine_null`;
it's interference/capacity-limited not retention-limited: `neuro_decay_gate_saturation_null`).

It is **NOT** a replacement for GDN-2 and **NOT** softmax attention. GDN-2's
gated delta rule stays; log-linear changes only how state is *accumulated across
chunks and read back*. Per arXiv:2506.04761 §5 it composes with Gated DeltaNet as
an orthogonal enhancement. Stays sub-quadratic: **O(n log n)**.

## 2. Why now vs. why defer

**The pull:** it's the only on-target attack on the recall ceiling. Everything
cheaper has been tried and capped (split-replay already gives the 2-4x it can;
decay-gate desaturation gave zero; more budget gave zero).

**The push to DEFER the build:** this is a **heavy, numerically-dangerous kernel
project**, not a quick screen — and this project has a graveyard of ambitious
attempts (~11 nulls). Two concrete reasons it's risky here specifically:
1. **Numerical stability.** `_gdn2_dense_chunk` ([src/delta_chunked.py:355-404](../src/delta_chunked.py#L355))
   already required a bespoke two-level GLA factorization to keep every exponential
   `<= 1` — a prior midpoint-shift factorization "overflowed/zeroed real
   contributions for trained fast-decay channels (alpha ~1e-6/step)". A hierarchical
   combine multiplies MORE decay factors across levels; getting it overflow-free is
   the hard part, not the plumbing.
2. **VRAM (measured estimate).** One state `S` at B=18, K=V=1280, fp32 ≈ **118 MB**.
   The hierarchy keeps L≈log2(512)=9 levels **simultaneously** during the forward:
   ~1 GB/layer × 8 delta layers ≈ **8.5 GB added**. The flagship already ran at
   ~29 GB / 32 GB at B=18 (`neuro_flagship_b18_sizing`) — this **OOMs**. Mitigations
   (bf16 states → ~4 GB, coarser chunk-level hierarchy L≈3 → ~3 GB, log-linear on a
   subset of layers, or lower B) all exist but each is a knob that must be tuned and
   re-validated. VRAM is a first-class design constraint here, not an afterthought.

**Conclusion:** the immediate next flagship (v2) should NOT wait on this. Do
Tier-1 (more tokens + curriculum + fair LR) now — reliable win, zero kernel risk.
Run log-linear as a parallel R&D thread gated on a de-risking spike.

## 3. Where it plugs into the actual codebase

| Surface | file:line | What changes |
|---|---|---|
| State carry (the core change) | [src/delta_chunked.py:475-485](../src/delta_chunked.py#L475) `gdn2_chunked_dense` | The single `S` carried across the chunk loop becomes L hierarchical states; add a `gdn2_loglinear_chunked` **parallel kernel** (do NOT mutate the parity-tested `gdn2_chunked_dense` — memory `neuro_gdn2_dense_chunked`) |
| Per-chunk primitive | [src/delta_chunked.py:355-404](../src/delta_chunked.py#L355) `_gdn2_dense_chunk` | Reusable for intra-chunk math; the cross-chunk term (the `S` read/write) is what the hierarchy replaces |
| Call site | [src/model.py:650](../src/model.py#L650) `SpikingHebbianBlock.forward_sequence` | Dispatch to the new kernel behind a config flag `gdn2_loglinear: true` (default off, ckpt-compat preserved) |
| Config plumbing | [src/train_gpu.py](../src/train_gpu.py) `build_model` | `gdn2_loglinear=c.get("gdn2_loglinear", False)` + level-count / dtype knobs |
| State allocator | [src/model.py:1104,1166](../src/model.py#L1104) | `M`-state shape becomes (L, B, K, V); brain-save/load must handle the hierarchy (the persistent-memory artifact grows log×, still portable — preserves the differentiator, unlike HOLA's KV cache) |

**Constraints the design must respect:** (a) autocast is hard-disabled inside the
kernel (bf16 triangular-inverse NaN'd training in 100 steps, 2026-07-15) — the
hierarchical combine inherits this fp32-internal rule; (b) the fla Triton kernel
was rejected because recall collapses at H>=8 (`neuro_gdn2_dense_chunked`) — the
port target is the paper's kernel adapted to OUR dense path, not fla.

## 4. Cost & risk

- **Params:** ~zero new params (per-level read weights are tiny) — it's a state/compute change, not capacity.
- **Compute:** +O(log n) read factor; the chunk GEMMs dominate, so throughput hit is likely modest (needs measurement — the spike answers this).
- **VRAM:** the dominant risk, ~+3-8.5 GB depending on level count / dtype (see §2). At B=18 the naive version OOMs.
- **Failure modes:** (1) numerical — silent NaN/zeroing of fast-decay channels, the exact class already hit once; (2) recall no-op — the hierarchy trains but N>=32 recall doesn't move (would mean the ceiling isn't state-resolution but something else); (3) wasted GPU — a full 90M screen is ~8h/arm, so a failed screen is a real cost; (4) confound — must NOT be screened concurrently on a GPU running another job (share-VRAM contention skews throughput/quality numbers).

## 5. Recommendation & decision gate

**Two-stage gate. Do NOT jump to a full integration.**

**Stage A — kernel de-risking spike (~1-2 days, 5080-eligible, NO flagship commit):**
Build `gdn2_loglinear_chunked` for a SINGLE layer at small shape (30M, B=8,
block_size=512). Success = (i) forward+backward parity vs a slow reference
(the house rule: every new kernel gets a `temp/test_*` parity test vs a sequential
oracle, as `gdn2_chunked_dense` did); (ii) no NaN over 500 training steps incl.
fast-decay channels; (iii) VRAM + throughput measured at 30M so the 363M budget is
known. **If the combine can't be made overflow-free or throughput craters, STOP and
pivot to HOLA (§ below) — that's a valid outcome.**

**Stage B — screened A/B (only if A passes):** 30M 2-seed, control = current
composed recipe, treatment = +log-linear on the delta layers. **Pre-registered win
(from Step-3 research): `recall_eval --replay split` improves at N=32 AND N=64 by
more than the 2-seed spread, AND val_ppl within seed noise, AND throughput regression
< 25%.** Recall measured with `--replay split` ONLY (single-presentation reads
near-chance here — `neuro_flagship_363m_outcome`). Winner → 90M 2-seed → flagship-v3.

**HOLA alternative** ([arXiv:2607.02303](https://arxiv.org/abs/2607.02303)): a
bounded exact-KV cache alongside the delta state. Simpler to build (no hierarchical
combine) but adds a growing exact-memory component that **breaks the portable
fixed-size-state differentiator** (a KV cache is not the project's 17 KB saveable
matrix). Keep as the fallback if Stage A shows log-linear's kernel is intractable;
do not lead with it — it's off-mission.

## 6. References

1. **Log-Linear Attention** — [arXiv:2506.04761](https://arxiv.org/pdf/2506.04761)
   (ICLR 2026), code **github.com/HanGuo97/log-linear-attention**. Primary port
   target: the O(log n) hierarchical-state mechanism + Triton kernels; §5 shows
   composition with Gated DeltaNet. **Port the state-hierarchy math; adapt to our
   dense fp32 chunk path, not fla.**
2. **GDN-2** — [arXiv:2605.22791](https://arxiv.org/pdf/2605.22791). The memory rule
   log-linear enhances; our impl is `gdn2_chunked_dense`.
3. **Eval rigor** — recall measured `--replay split` per this project's finding that
   single-presentation reads near-chance (`neuro_flagship_363m_outcome`); 2-seed
   screen protocol per `neuro_composed_recipe_confirmed_2026_08_07`. Recurrent-model
   AR is LR-sensitive ([arXiv:2508.19029](https://arxiv.org/html/2508.19029v2)) — hold
   LR fixed across arms or the recall delta is confounded.
4. **HOLA** (fallback) — [arXiv:2607.02303](https://arxiv.org/abs/2607.02303).
5. Lineage: log-linear is the "middle ground between linear and softmax" for the
   fixed-state recall limit; see also Sparse State Expansion (arXiv:2507.16577).
