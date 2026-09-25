# Decay-Gate Saturation — Design Doc

**Date:** 2026-08-17
**Status:** **Phase 0a RUN — result is NULL. Phase 1 is NOT recommended.** See §0 below.
~~Scoped, not built. Recommendation: build — but run the LR-sensitivity control FIRST~~ (superseded by the Phase 0a result).

---

## 0. RESULT — Phase 0a (2026-08-17): decay-gate saturation is NOT the cause

Ran the inference-only `f_logit` floor sweep on `step_0124000.pt`
(`temp/phase0a_clamp_sweep.py`, log `logs/phase0a_clamp_sweep.log`).
Instrument: default-off `gdn2_f_floor` clamp added at [src/model.py:626](../src/model.py#L626).

**Controls both passed.** The `floor=None` arm reproduced the known baseline
100/100/97/50/22 exactly (identity check), and the clamp verifiably fired —
`decay_sat_lo` went from 99%/35%/42%/16% (blk0/6/7/8) to **0% on every block**.

MQAR among-candidates accuracy, `--replay split`, B=32:

| f_logit floor | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| none (control) | 100% | 100% | 97% | 50% | 22% |
| **−4.0** (fully desaturates) | 100% | 100% | 97% | **50%** | **22%** |
| −3.0 | 100% | 100% | 97% | 53% | 12% |
| −2.0 | 100% | 100% | 100% | 25% | 0% |
| −1.0 | 100% | 94% | 59% | 6% | 0% |

**Findings:**
1. **Fully removing never-forget saturation changes recall by exactly zero.**
2. **Every step toward more forgetting monotonically destroys recall.** The
   never-forget behavior is functionally *load-bearing*, not pathological.
3. **Recall is not retention-limited.** At floor −4.0 blk0's mean α falls
   0.998 → 0.992; compounded over a 128-token N=64 sequence that cuts retained
   signal from ~77% to ~36% — and recall did not move at all. Whatever caps
   high-N recall here is interference/capacity, not memory decay.

**Consequence for Phase 1: do not run it as designed.** All four candidate arms
(entropy penalty, cap, gate LR, ReGLA refining gate) push gates *away* from
never-forget — the exact direction just measured as neutral-to-harmful.

**Honest limitation.** This is an inference-time intervention on a model *trained*
with saturated gates. It shows the saturated state is not hurting this trained
model's recall; it does not strictly exclude that a model trained from scratch
without saturation would have found a better solution. That residual question is
much more expensive to answer (a full training arm) and finding 2 argues against
it being worth the GPU time: the direction of travel is wrong.

**Where the recall regression actually likely comes from.** With saturation
excluded, the live hypothesis is the **extreme-N-recall vs general-LM tradeoff**
(§2 rival explanation 1): capacity reallocated away from synthetic high-N recall
as real-language training dominated. Not directly testable on current artifacts —
recall is flat across every checkpoint we have (112k/124k/132k ≈ 53/50/56 at
N=32), and the decline happened between 200M and 4.1B tokens where the only
checkpoint (the shakedown) was deleted. Testing it needs fresh mid-training
checkpoints from a new run, not a fix to the existing one.

---
**Related:** memory `neuro_flagship_363m_outcome`, `neuro_flagship_neverforget_saturation`, `neuro_composed_recipe_confirmed_2026_08_07`, `neuro_gdn2_osdn`

---

## 1. What it is

A set of A/B experiments to stop the GDN-2 **channel-wise decay gate** from saturating into a
permanent never-forget state, and to test whether fixing that recovers the high-N associative
recall the flagship lost.

This is **not** an architecture redesign and **not** a replacement for GDN-2. It is a targeted
regularization/optimization fix to one gate, tested at 30M and promoted to 90M under the existing
2-seed protocol. Scope is deliberately narrow: 3-4 candidate interventions, one screen, one
promotion gate.

**Important correction to the original framing.** The pre-existing sketch in memory
`neuro_flagship_neverforget_saturation` named `W_alpha` as the target of all three candidate
fixes. **That is the wrong parameter.** Under the GDN-2 recipe the flagship actually ran,
`W_alpha` is not the active decay lever at all — the `decay_gate` branch that uses it
([src/model.py:343-349](../src/model.py#L343), [:674](../src/model.py#L674)) is a *different*
code path. The saturating gate is `W_fdecay` ([src/model.py:285](../src/model.py#L285)), whose
logit feeds:

```python
f_logit   = self.W_fdecay(spk_seq)                          # model.py:615
if self.gdn2_gate_norm:
    f_logit = self.gate_norm_f(f_logit) + self.gate_bias_f  # model.py:619
alpha_seq = torch.exp(-torch.exp(self.gdn2_a)
                      * F.softplus(f_logit))                # model.py:622-623
```

So `α = exp(−exp(gdn2_a) · softplus(f_logit))`, and the reported `d=neverforget` statistic is
`decay_sat_lo` = fraction of `f_logit < −4.595` ([src/model.py:129](../src/model.py#L129)).
`f_logit → −∞` ⟹ `softplus → 0` ⟹ `α → 1` ⟹ channel never forgets. **Any fix must act on
`f_logit` / `gdn2_a` / `gate_bias_f`, not `W_alpha`.** Writing the fix against `W_alpha` would
have produced a silent no-op A/B — exactly the "silently different arch" bug class this project
has already been bitten by four times (memory `neuro_train_sft_build_drift_fix`).

## 2. Why now vs. why defer

**Why now:**
- The 5090 and 5080 are both **free** (flagship killed 2026-08-17 at step 135k).
- This is the top Open item in `to-do.html` and the designated next-recipe work.
- It is now backed by a *measured* cost, not just telemetry aesthetics: split-replay recall at
  N=32/64 is ~53/22% against the 200M-token shakedown's 94/63% — and that gap is a **whole-run
  property** (already present at step 112k), not late-run decay.

**Honest caveat on motivation.** The recall regression is *correlated* with gate saturation, not
yet shown to be *caused* by it. Two rival explanations are live and are not yet excluded:
1. **Extreme-N recall vs. general-LM tradeoff** — the shakedown was at 200M tokens; by 4.6B the
   model may simply have reallocated capacity from synthetic high-N recall to real language.
   Nothing in our data separates this from the saturation story.
2. **Depth pattern may be normal.** The literature reports the *opposite* depth ordering from
   ours as healthy: gate distributions "shift toward values close to 1.0" as depth *increases*,
   with shallow layers doing the forgetting (§6, CAWN/GatedFWA). Our blk0 (shallowest, 99%
   pinned) is the anomaly; our blk6/7/8 drift may be the *expected* healthy pattern. A fix that
   forces deep-layer gates open could therefore be actively harmful.

This is why §5 gates the work behind a causal check rather than jumping to the fix.

**Why not defer:** the GPUs are idle and the next flagship should not be launched on a recipe with
a known-unexplained memory pathology. The cost of scoping it now is low; the cost of another
6-day run that regresses recall is high.

## 3. Where it plugs into the actual codebase

| What | File:line | Note |
|---|---|---|
| `W_fdecay` definition + init (`bias=-3.0`, `weight*0.01`) | [src/model.py:285-291](../src/model.py#L285) | The deliberate slow-decay init |
| `gdn2_a` learnable decay scale (`d_mem`, zeros init) | [src/model.py:286](../src/model.py#L286) | `exp(a)=1` at init; a second saturation lever |
| `gate_norm_f` RMSNorm + `gate_bias_f` | [src/model.py:304-308](../src/model.py#L304) | Gate-norm carries the −3.0 offset explicitly |
| **Gate forward (the intervention point)** | [src/model.py:613-623](../src/model.py#L613) | `f_logit` → `softplus` → `alpha_seq` |
| Saturation telemetry `_gate_sat_stats` | [src/model.py:105-132](../src/model.py#L105) | Already reports signed `decay_sat_lo/hi` + `decay_std` |
| α-space telemetry `_alpha_stats` | [src/model.py:170-185](../src/model.py#L170) | Already wired for GDN-2 at [:635](../src/model.py#L635) |
| Telemetry print | [src/train_gpu.py:800](../src/train_gpu.py#L800) | `[gates sat]` line |
| **Aux-loss attach point** | [src/train_gpu.py:573/632/641/653/719](../src/train_gpu.py#L637) | Established pattern: `main_loss = main_loss + w * term` |
| Optimizer param routing (Muon vs AdamW, wd split) | [src/muon.py:226-235](../src/muon.py#L226) | Already special-cases `gdn2_a`/`gate_bias_f` as 1D no-decay |

**Constraints already baked in that the design must respect:**
- `W_fdecay`'s `bias=-3.0` init is *deliberate* (slow decay ≈ `lam≈0.98` behavior). Under
  `gdn2_gate_norm: true` the RMSNorm would wash it out, so it is carried in `gate_bias_f`
  instead. **Any cap/penalty must not fight this init** — pushing gates open at step 0 would
  undo a decision that is already validated in the composed recipe.
- Muon's NS iteration degenerates on 1-row matrices; `muon_precise_gate_filter` /
  `muon_fix_gate_matrices` exist for this (memory `neuro_muon_gate_matrix_bug`). A new
  per-gate optimizer group **must** route through the same guard.
- `gdn2_a` and `gate_bias_f` are already excluded from weight decay ([muon.py:229](../src/muon.py#L229)).
  A separate-LR group must preserve that.
- Telemetry needed to judge the fix **already exists** — no new instrumentation required.

## 4. Cost & risk

**Compute.** 30M screen ≈ hours/arm; 90M promotion ≈ ~8h/arm (per prior screens in `logs/`).
With 2 GPUs and the arms below, the full program is ~2-4 days wall-clock. Cheap relative to a
6-day flagship.

**Params/memory.** Candidates (a) and (b) add **zero** parameters. Candidate (d) ReGLA-style
refining gate adds one `Linear(n_neurons, d_mem)` per block — non-trivial at flagship width, and
that cost must be counted against any win.

**Failure modes:**
- **Wrong-parameter no-op** (patching `W_alpha`): mitigated by §1 — but this is the single
  highest-probability failure and must be verified by an identity/telemetry check before any arm runs.
- **LR confound (most dangerous).** Candidate (c) *is* an LR change. arXiv:2508.19029 shows
  recurrent models on AR have a razor-thin LR window — "performance remains near zero for most
  learning rates but suddenly reaches near-optimal levels at specific values." A (c) win could be
  pure LR luck, not desaturation. Any (c) result is uninterpretable without an LR control.
- **Fixing a non-problem.** If the deep-layer drift is the healthy depth pattern (§2), forcing
  those gates open costs quality. Mitigated by per-block reporting, not just aggregates.
- **Confound with in-flight work:** none — GPUs are idle, nothing else running.
- **Noise floor.** This project has already closed one enhancement as null when the effect sat
  under the seed noise floor (memory `neuro_moe_phase1`). 2 seeds is the established minimum and
  seed-to-seed spread has been as high as 90% pre-QK-norm (memory `neuro_qknorm_fix_confirmed_2026_08_02`).

## 5. Recommendation & decision gate

**Recommendation: build — but run Phase 0 first, and let it decide whether Phase 1 happens at all.**

The causal question ("does saturation actually cost recall?") is cheap to answer and, if the
answer is no, it saves the entire A/B program. Do not start with the fixes.

### Phase 0 — causal check + LR control (do this first, ~1 day)

**0a. Does desaturation recover recall? (intervention-free test.)** Take the existing flagship
checkpoint `step_0124000.pt` and, at *inference only*, clamp `f_logit` to a floor (e.g. −4.0,
−2.0) to force the pinned channels back into an adaptive range. Re-run
`recall_eval --replay split --B 32`. No training needed.
- Recall improves ⟹ saturation is *causally* implicated → proceed to Phase 1.
- Recall unchanged/worse ⟹ saturation is a symptom, not the cause → **stop; write it up as a
  null and redirect to the recall-vs-LM-tradeoff hypothesis instead.**

**0b. LR-sensitivity baseline.** Per arXiv:2508.19029, sweep the *unmodified* 30M recipe across
≥4 LRs, 2 seeds, recording recall. This establishes how much recall moves from LR alone. **Any
Phase 1 effect smaller than this band is not a real effect.** This is the pre-registered noise
floor and it is what makes candidate (c) interpretable.

### Phase 1 — arms (only if 0a is positive)

Ranked by evidence strength, best first:

| Arm | Mechanism | Params | Why |
|---|---|---|---|
| **(b) decay floor / cap** | clamp `f_logit ≥ f_min` (equivalently cap α ≤ α_max) at [model.py:622](../src/model.py#L622) | 0 | Directly matches FG²-GDN's "gate clamping"; simplest possible change; one line |
| **(d) ReGLA refining gate** | `F = (1−r)⊙g² + r⊙(1−(1−g)²)` | +1 Linear/block | Only method whose *stated purpose* is restoring gradient in the saturated region; strongest mechanistic rationale |
| **(a) entropy/diversity penalty** | `main_loss + λ·penalty(f_logit)` at the [train_gpu.py:641](../src/train_gpu.py#L637) pattern | 0 | Cheap; but a blunt instrument — pushes *all* channels off the rail, including ones legitimately pinned |
| **(c) separate gate LR** | own optimizer group for `W_fdecay`/`gdn2_a`/`gate_bias_f` | 0 | **Run last, and only with the 0b control.** Confounded by construction |

Protocol: 30M 2-seed screen → winner to 90M 2-seed (the protocol that vetted the composed recipe,
memory `neuro_composed_recipe_confirmed_2026_08_07`).

### Pre-registered success criterion

An arm **wins** only if, at 30M, 2 seeds, versus the unmodified control:
1. **`recall_eval --replay split`** improves at **N=32 and N=64** by more than the Phase-0b LR
   band **and** more than the seed spread; **and**
2. **val_ppl does not regress** beyond seed noise; **and**
3. `decay_sat_lo` is measurably reduced on the affected blocks (confirms the mechanism actually
   fired — guards against a win-by-accident).

All three, or it does not promote. Recall is measured with **`--replay split`** — single-presentation
reads near-chance at N≥32 on this model and would make every arm look identical
(memory `neuro_flagship_363m_outcome`).

**Explicitly not a win:** val_ppl improvement alone. The flagship already demonstrated that
val_ppl and recall are decoupled here; a ppl-only gain does not address the problem this doc exists to fix.

## 6. References

**Primary port targets**
1. **FG²-GDN — "Enhancing Long-Context Gated Delta Networks with Doubly Fine-Grained Control"**,
   [arXiv:2604.19021](https://arxiv.org/pdf/2604.19021). *Closest match to our exact pathology on
   our exact architecture:* identifies GDN forget gates that "pinch to near-unity, essentially
   disabling the gating mechanism." Uses gate clamping + separate gate LR schedule + anti-collapse
   init — i.e. our candidates (b) and (c), already validated by someone else on GDN. **Port the
   clamping scheme from here.**
2. **ReGLA — "Refining Gated Linear Attention"**,
   [arXiv:2502.01578](https://arxiv.org/html/2502.01578v1). The gradient analysis:
   `∇g = g⊙(1−g)` → vanishing gradient at the rails, gate freezes. Fix, verbatim:
   `Fₜ = ((1−rₜ)⊙gₜ² + rₜ⊙(1−(1−gₜ)²))1ᵀ`, with `gₜ = σ(Wₘx+bₘ)`, `rₜ = σ(Wᵣx+bᵣ)`.
   Their Figure 4 (vanilla GLA gates stuck at extremes vs. ReGLA's diverse distribution) is
   **the exact plot we should reproduce** from our `decay_std` telemetry.

**Eval protocol / rigor**
3. **"Revisiting associative recall in modern recurrent models"**,
   [arXiv:2508.19029](https://arxiv.org/html/2508.19029v2). **The most important methodological
   reference here.** Recurrent models on AR succeed only in an extremely narrow LR window;
   prior literature systematically under-reported recurrent performance by not sweeping LR.
   Mandates per-arm LR sweeps and reports 3 seeds (42/123/777). This is the basis for Phase 0b.
4. **Zoology — "Measuring and Improving Recall in Efficient Language Models"**,
   [arXiv:2312.04927](https://arxiv.org/abs/2312.04927). Origin of MQAR; 82% of the
   attention-vs-gated-conv perplexity gap is attributable to in-context recall. Also the warning
   that state size dominates recall (a 70M attention model beats a 1.4B gated-conv model), so
   recall comparisons must hold state size fixed.

**Lineage / context**
5. **GDN-2 — "Decoupling Erase and Write in Linear Attention"**,
   [arXiv:2605.22791](https://arxiv.org/pdf/2605.22791) — the architecture we run.
6. **GLA — "Gated Linear Attention Transformers with Hardware-Efficient Training"**,
   [arXiv:2312.06635](https://arxiv.org/pdf/2312.06635) — decay term ≙ Frobenius-norm
   regularization on memory; useful framing for why an always-1 gate degenerates to an
   unregularized accumulator.
7. Depth-pattern counter-evidence (gates → 1.0 in *deeper* layers is reported as normal):
   see CAWN [arXiv:2604.04250](https://arxiv.org/html/2604.04250) and GatedFWA
   [arXiv:2512.07782](https://arxiv.org/pdf/2512.07782). **Read before assuming blk6/7/8 drift is a bug.**
