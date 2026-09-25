# Improvement Plan — Pre-Flagship-v2 (2026-08-27)

Research synthesis + ranked, testable improvement shortlist for the next flagship,
following the "matched baseline → improve → re-baseline" sequence. Grounded in
this project's actual history (what's been tried, what's null) and a 2026 SOTA
research pass. **Deliberately biased away from bolt-on module search** — the
project has ~11 null/negative module results (MoE, fracconn, SDM, adaptive-compute,
PMA, MQAR-refine, sparsity-refine, DPO×3, decay-gate, z_loss). The wins below are
either known-good levers or targeted at the ONE measured architectural weakness.

## The one measured weakness worth targeting

**High-N associative recall is capped, and it's architectural, not budget.** Proven
twice: [[neuro-500m-post-pretrain-refine-null]] (4x budget didn't move it) and
[[neuro-decay-gate-saturation-null]] (recall is interference/capacity-limited, not
retention-limited). The flagship sits at split-replay ~53/22% at N=32/64. This is
the fixed-size-state limitation of linear attention — a known, named problem with
2026 solutions (Tier 2 below).

---

## Tier 1 — Known-good, low-risk. Fold into flagship-v2 regardless.

These need no research bet; they're near-certain wins we simply haven't done.

1. **Train on more tokens.** The corpus has **14.5B prepared** tokens
   (`data/tokenized/meta.json`); the flagship used **4.79B** (33%). At 363M that's
   ~13 tok/param — below Chinchilla-optimal (20x ≈ 7.3B) and far below available.
   The single biggest quality lever, zero research risk. **flagship-v2: 10-14.5B tokens.**
   The divergence guards shipped 2026-08-25 ([[neuro-flagship-divergence-root-cause]])
   make longer runs safe (they'd have caught the head-norm blowup at step 130k).

2. **Enable the context curriculum.** `meta.json` defines
   `curriculum_milestones_tokens` (blk512→3.3B, blk1024→9.9B, blk2048→14.5B) — a
   designed-but-unused schedule. Longer context late in training improves
   long-range modeling without extra data. Verify the loader honors it (it was
   `curriculum_enabled: false` on the flagship).

3. **Fair / transferable LR.** Do NOT reuse a single tuned LR across scales blindly.
   Recent finding: **weight decay may matter more than μP for LR transfer**
   ([arXiv:2510.19093](https://arxiv.org/pdf/2510.19093)) — much cheaper than
   implementing full μP (which also *breaks* with trainable norm gains, which we
   have). Pair with a token-count-agnostic schedule (**Power scheduler**,
   [arXiv:2408.13359](https://arxiv.org/pdf/2408.13359), or WSD) so budget can scale
   without re-tuning. The transformer-baseline LR sweep running now
   (`screen_tfmr_lr_*`) is the first instance of taking LR fairness seriously.

## Tier 2 — Targeted at the recall ceiling. Higher value, real cost. SCREEN FIRST.

4. **Log-Linear Attention** ([arXiv:2506.04761](https://arxiv.org/pdf/2506.04761),
   ICLR 2026, code: github.com/HanGuo97/log-linear-attention). Replaces the
   fixed O(d) state with an **O(log n)** state — directly lifts the recall ceiling
   that we proved is architectural. **Composes with Gated DeltaNet** (our exact
   memory rule) as an orthogonal enhancement, not a replacement. Stays sub-quadratic
   (O(n log n)), has Triton kernels. **Cost:** "moderately invasive" to our
   `gdn2_chunked_dense` kernel ([[neuro-gdn2-dense-chunked]]) — the hardest part.
   This is the highest-upside on-target bet. Scope as its own design doc before building.

5. **HOLA-style exact-KV hybrid** ([arXiv:2607.02303](https://arxiv.org/abs/2607.02303)).
   Keeps the delta-rule state as compressive memory + adds a **bounded exact KV
   cache** for what the recurrent state forgets. Alternative attack on the same
   ceiling; more memory cost but conceptually simpler than log-linear. A fallback
   if log-linear's kernel work proves too invasive.

## Tier 3 — Sanity checks (cheap, do opportunistically)

6. **Confirm GDN-2 is still the right memory rule.** 2026 frontier added KDA (Kimi
   Delta Attention), Mamba-3, OSDN, Erase-then-Delta ([arXiv:2606.26560]). We chose
   GDN-2 over SDM in July; a quick lit-check that nothing has clearly superseded it
   for the persistent-memory (split-replay) regime is worth an hour.
7. **z_loss factorial** (from [[neuro-zloss-ab-2026-08-26-null]]) — the combined
   1e-3+include_mtp hurt; a 2x2 isolates whether either knob alone helps the
   divergence guard without the val_ppl cost.

---

## Recommended sequence

- **Now (in motion):** transformer LR sweep → matched-transformer baseline vs the
  current flagship (measurement: does our arch still win at 363M as it did at 20M?).
- **flagship-v2 (the boring, reliable next run):** current validated recipe
  ([[neuro-composed-recipe-confirmed-2026-08-07]]) + **Tier 1** (more tokens +
  curriculum + fair LR + shipped guards). Near-certain improvement over the current
  flagship, near-zero research risk. This should be the default next flagship.
- **In parallel, screen Tier 2** (log-linear attention at 30M/90M, same 2-seed
  protocol as the composed recipe). If it lifts high-N recall without a val_ppl or
  throughput regression, fold into a **flagship-v3**.
- **Re-baseline** the improved model against a fresh matched transformer (per the
  user's plan — always compare against our best).

The discipline that makes this "optimized" and not more lottery: **Tier 1 ships
without a bet; Tier 2 must pass a pre-registered screen before touching a flagship.**
