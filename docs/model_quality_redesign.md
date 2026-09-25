# Why the model still produces nonsensical output — diagnosis + redesign plan

Date: 2026-07-31, updated 2026-08-01 with overnight de-risking results.
Status: diagnosis complete, plan drafted and checked against 2025–2026
published research, P2/P3/P5 changes implemented and screened at 30M+90M
with multi-seed confirmation. **Supersedes the "same recipe, bigger" premise
of `1b_v1_design.md`** — the recipe itself had three coherence-limiting
problems that scaling to 1B would NOT have fixed on its own.

This doc answers one question: *what will it actually take to make this novel
spiking + Hebbian model produce stellar responses?*

## 2026-08-01 overnight update — what shipped, what was screened, what's still open

**Shipped (code, in `src/hybrid_attn.py` + `src/model.py` + `src/train_gpu.py`):**
`HybridAttentionBlock` (full softmax MHA + RoPE + SwiGLU), wired via
`attention_layer_positions` config list — a clean drop-in matching
`SpikingHebbianBlock`'s `forward_sequence` interface, verified byte-identical
when unset (T1 backwards-compat test) and functionally correct with
attention on (build/forward/backward/state-save-load, `temp/test_hybrid_attn.py`).

**Built but NOT adopted:** a proper 32k BPE tokenizer trained from scratch on
the corpus (`data/tok_32k/`) — technically zero-drop, but incompatible with
KD (different token boundaries than Llama-3). Switched to a **top-32k
most-frequent Llama-3 subset** vocab instead (`vocab_map_32768.npy`,
auto-built by `StreamText`) — restores KD-compatibility, drops only ~2-3%
of tokens (vs the old 16k LUT's ~10%) instead of the from-scratch tokenizer's
0%. This is the vocab actually screened below and recommended for the real
pretrain.

**KD (P4) — tested, REJECTED for now.** Three 30M A/Bs against a matched
no-KD control (`meta-llama/Llama-3.2-1B` teacher, top-32k-Llama3-subset
vocab so teacher/student share index space):

| Config | val_ppl @500/1000/1500 | vs control |
|---|---|---|
| Control (no KD) | 320 / 183 / 142 | — |
| KD alpha=0.5 | 345 / 215 / 177 | +8% → +18% → +24% worse |
| KD alpha=0.2 | 325 / 189 / 150 | +1% → +4% → +5% worse |

Gap widens within each run at every alpha tested — the signature of
"hard-CE is straightforwardly better here," not "KD needs more tokens to pay
off" (which would show a shrinking gap). Did not proceed to a 90M KD retest;
the consistent, worsening-not-improving trend across three alpha values
didn't support the hypothesis. **KD is off the table for 1B v1 unless
revisited with a different mechanism (e.g. much lower alpha, or a
sequence-level rather than distributional KD loss).**

**Sparsity target (P5) — 30M elasticity curve mapped**, not yet chosen for
the real run:

| λ | spike @1500 (final) | val_ppl @1500 | memory Δ @1500 |
|---|---|---|---|
| 0.3 (baseline) | 0.216 | 142 | +225 |
| 0.5 | 0.155 | 265 | +209 |
| 0.7 | 0.150 | 287 | +194 |

λ=0.5 already gets spike to ~target (0.15-0.16) with memory-Δ holding up
reasonably (+209 vs baseline's +225); λ=0.7 buys almost nothing further on
spike (0.150 vs 0.155) for a real val_ppl cost. **Recommend λ=0.5 for the
1B run**, watching memory-Δ as the tripwire (matches the project's existing
`sparsity_lambda` tuning discipline). Not fully settled — this sweep ran
concurrently with the seed-variance investigation below and inherited its
confounds (see caveats in the sweep configs); a same-GPU, multi-seed
elasticity confirm is warranted before locking λ for the real pretrain.

**Seed-variance investigation — the major finding of the night.** A routine
seed1 reconfirm (this project's standing 3-strikes rule: MoE, Engram v1, and
uniform recurrent-depth all reversed on reseed) found a large, real, and
initially concerning divergence at 30M:

| Step | Seed0 | Seed1 | Seed2 |
|---|---|---|---|
| 500 | 320 | 438 | 440 |
| 1000 | 183 | 314 | 312 |
| 1500 | 142 | 269 | 268 |

**Seed1 and seed2 track each other almost exactly (within 1%) at every
point; seed0 is the outlier**, not the other way around as first assumed.
Confirmed real (not a GPU/batch confound) via a same-5090, same-batch
seed1 re-run matching the confounded 5080 run to within 0.6% at every eval
point. Confirmed NOT caused by the new hybrid attention layer via a full
2×2 (seed × attention) ablation:

| | Seed0 | Seed1 |
|---|---|---|
| With attention | 142 | 269 |
| Without attention | 287 | 287 |

Without attention, both seeds converge to nearly the SAME (worse) value —
attention gives seed0 a real, substantial average benefit (~2x val_ppl
improvement) that seed1 largely fails to capture. **Attention helps on
average; its benefit is itself seed-dependent, and at 30M the "typical" seed
(2 of 3 sampled) doesn't get much of that benefit.**

**At 90M scale, the effect shrinks substantially but does not fully resolve**
by the end of a 90M-token budget. Full 3-seed table (all runs complete):

| Step | Seed0 | Seed1 | Seed2 |
|---|---|---|---|
| 1000 | 232 | 182 | 181 |
| 2000 | 126 | 139 | 141 |
| 3000 | 93 | 118 | 118 |
| 4000 | 75 | 106 | 107 |
| 5000 | 65 | 98 | 99 |
| 6000 | 59 | 93 | 94 |
| 7000 (final) | **56** | **91** | **91** |

**Seed1 and seed2 track each other within 1-2% at every single point across
both scales — this is the same "2 of 3 seeds agree, seed0 is the outlier"
pattern as 30M, now confirmed at 90M too.** Final gap (seed0 vs seed1/seed2):
62-63% worse, essentially identical between seed1 and seed2. The gap-growth
rate is decelerating (increments per 1000-step eval: 31→17→14→9→8→4
percentage points) — approaching a plateau, not a runaway divergence — but
has NOT closed within a 90M-token budget. β telemetry stayed completely
healthy in every seed at every scale (μ 0.49-0.69, 0% dead, 0% at_floor) —
this rules out beta-collapse as the mechanism; the actual cause
(embedding/output-head init interaction with the delta-rule memory pathway?
NorMuon Newton-Schulz sensitivity?) is unidentified.

**This is now a well-established, 3-seed-confirmed finding, not a
single-seed scare**: at both 30M and 90M, 2 of 3 seeds show substantially
worse convergence than the "lucky" seed0, and the gap — while shrinking in
relative terms with scale — remains large (60%+) at 90M. Do not extrapolate
that this closes by 500M or 1B without direct evidence.

---

## 2026-08-01 — mechanism analysis + path to "ready to scale"

### What the variance is, precisely (localized to attention)

The cleanest fact in all the data: **without attention, the recipe is
seed-robust; with attention, it is not.** The two no-attention runs (seed0,
seed1) land at 449/328/287 and 450/328/287 — nearly bit-identical across the
whole trajectory. The three with-attention runs diverge (142 vs 268/269 final
at 30M). So the spiking + Hebbian core is stable across seeds; **the added
attention layer is the sole source of the seed variance.** This matches
published findings: attention-logit growth is a known training instability,
and *middle-layer* attention (ours sits mid-stack, layer 2/4 and 3/6) is the
least seed-stable part of a transformer.

### What it is NOT (ruled out)

- **NOT beta-collapse / memory-pathway failure** — β telemetry healthy in
  every seed at every scale (0% dead, 0% at_floor).
- **NOT trivial component-collapse** (attention weights vanishing) —
  `temp/diag_attention_utilization.py` measured the attention layer's output
  magnitude on seed0 (lucky) vs seed2 (typical) 90M checkpoints:
  attn/residual norm ratio 1.03 vs 1.11, o_proj weight |mean| 0.0401 vs
  0.0410 — essentially identical. Attention is *active* in both seeds; it is
  not switched off in the typical one. (Caveat: measured on random input, so
  this rules out zero-weight collapse but can't measure learned usefulness.)

So the mechanism is subtler than "attention turns on for one seed": attention
is active in all seeds but converges to a *more useful function* on the lucky
seed. Two live hypotheses, both addressed by the same next step:

- **H1 (instability):** attention training is unstable at this
  scale (unbounded logits without QK-norm; NorMuon's aggressive Newton-Schulz
  applied to the attention qkv/o_proj matrices — research flags AdamW as more
  seed-stable for attention). Fixable with standard stabilizers.
- **H2 (undertraining):** at <1.5 tok/param, attention is simply undertrained
  and seed-noisy, and would stabilize with the real (100B+) token budget.
  Plausible — the gap DID shrink 30M→90M — but cannot be assumed.

### The blocker for scaling, stated plainly

A 500M/1B run costs GPU-weeks and lands on *one* seed. With the current
recipe, that seed is a coin-flip between the seed0 trajectory (excellent) and
the seed1/seed2 trajectory (mediocre, barely better than no attention). **That
is the one thing that must be fixed before scaling** — not coherence (which
is a token-budget problem for later), but seed-robustness.

### Plan to get ready to scale (cheap, small-scale, multi-seed throughout)

**Step 1 — Stabilize the attention layer.** Screen these at 30M, each with
**3 seeds**, against the current (seed-fragile) recipe:
1. **QK-norm** (RMSNorm on Q and K before the dot product) — the standard,
   widely-adopted "free-lunch" fix (Qwen3, Gemma3, OLMo) for exactly this
   attention-logit-growth instability. Highest-probability single fix.
2. **Route attention matrices (qkv, o_proj) to AdamW** instead of NorMuon —
   research reports AdamW is substantially more seed-stable for attention
   heads. One-line change to `muon.build_muon_adamw`'s routing.
3. **Longer LR warmup** (currently ~2% of steps; try 5-10%) — cheap, helps
   unstable early training.
   Try QK-norm alone first (likely sufficient); stack the others only if
   needed. **Win condition:** the 3 seeds converge within noise (as the
   no-attn baseline already does) AND the mean beats both the no-attn 287
   and ideally approaches seed0's 142.

**Step 2 — Confirm at 90M** (3 seeds) once Step 1 clears at 30M. Same win
condition. Simultaneously lock the remaining knobs: sparsity λ=0.5
(multi-seed confirm), attention layer count/placement, vocab (top-32k subset,
already chosen).

**Step 3 — Mid-scale reality check.** One ~200-350M run before committing to
500M+, because the variance shrank-but-didn't-vanish 30M→90M — confirm it's
actually *gone* (not merely smaller) at a scale closer to target.

**Step 4 — Then scale**, with the real coherence lever from the original
diagnosis: 100B+ tokens (P1), the top-32k subset vocab, hybrid attention (now
stabilized), sparsity λ=0.5. Keep the persistent-Hebbian differentiator.

**Cross-cutting:** adopt **multi-seed screening as standing practice** — the
project's 3-strikes rule (MoE, Engram-v1, uniform-recurrent-depth) just became
4 near-misses; single-seed 30M "wins" are not trustworthy. And do the pending
code-hygiene pass (P6) to shrink the silent-bug surface before any multi-week
run.

**What this means for the 1B plan:**
- Do NOT treat the 30M/90M screens in this doc's earlier sections as
  "confirmed" on a single seed — they weren't, and the standing 3-strikes
  rule exists for exactly this reason.
- The redesigned recipe (32k-subset vocab + hybrid attention + relaxed
  sparsity) is NOT obviously broken — β is healthy, memory-Δ is positive
  and substantial in every seed, and the model trains stably — but it shows
  real seed-to-seed quality variance that shrinks with scale but doesn't
  vanish by 90M tokens.
- Before committing GPU-weeks to a 1B pretrain, this needs either (a) a
  larger-scale (500M-class) multi-seed check to confirm the gap keeps
  shrinking, or (b) root-causing the actual mechanism (candidate: NorMuon's
  Newton-Schulz on the embed/head matrices interacting unluckily with
  certain inits — untested) so it can be fixed rather than out-scaled.
  **Recommend (b) first** — it's cheaper than another multi-day multi-seed
  run and would directly derisk 1B regardless of what scale the effect
  fully vanishes at.

---

## 2026-08-02 — Step 1 + Step 2 CONFIRMED: QK-norm fixes the seed variance

**QK-norm implemented** (`src/hybrid_attn.py`: per-head `RMSNorm` on Q and K,
applied before RoPE; `attention_qk_norm` config flag, default on) and
screened exactly per the plan above.

**Step 1 (30M, 3 seeds, `configs/screen_qknorm_30m_s{0,1,2}.yaml`):**

| seed | val_ppl (final, step 1500) |
|---|---|
| 0 | 140.30 |
| 1 | 149.83 |
| 2 | 152.06 |

Mean **147.4**, spread **~8%** (140.30–152.06) — vs. the pre-QK-norm gap of
**~90%** (seed0=142 lucky outlier vs seed1/2≈269) and comfortably beating the
no-attn baseline of 287. Win condition met cleanly.

**Step 2 (90M, 3 seeds, `configs/screen_qknorm_90m_s{0,1,2}.yaml`, same
recipe scaled up per the existing `screen_new_recipe_v1_90m.yaml` shape:
d=640, n_layers=6, attention at position 3):**

| seed | val_ppl (final, 90M tok) |
|---|---|
| 0 | 54.76 |
| 1 | 52.52 |
| 2 | 58.53 |

Mean **55.27**, spread **~11%** (52.52–58.53) — vs. the pre-QK-norm 90M gap
of **~62-63%** (seed0=56 vs seed1/2=91). Notably, all three seeds now land
close to where the old "lucky" seed0 landed (56) — QK-norm doesn't cost the
best-case quality, it makes the typical case match it.

**Deliberate scope note:** both screens held `sparsity_lambda` at 0.3 (the
v2 control value) rather than the recommended 0.5, to isolate QK-norm as the
single changed variable. The λ=0.5 lock-in is still a follow-up, to be
combined with QK-norm rather than assumed compatible.

**Conclusion: H1 (attention training instability) was the correct
mechanism** — not H2 (undertraining). QK-norm alone, with no other change,
resolved the variance at both scales tested. Route-to-AdamW and longer-warmup
(the Step 1 fallbacks) were not needed.

**Status vs the plan:** Steps 1–2 done and clean. **Step 3 (mid-scale
~200-350M reality check) is next but not yet started** — it needs a new
architecture config (no existing validated shape at this size for the
hybrid+QK-norm+v2-vocab recipe, unlike Step 2 which reused
`screen_new_recipe_v1_90m.yaml`'s shape), so it's a real design decision
(width/depth/attention placement/token budget) rather than a mechanical
scale-up, and a good point to check in before committing more GPU-hours.

---

## The short version

The nonsensical output is **not** because the novel idea is bad. The Hebbian
memory is genuinely load-bearing (ablating it sends val_ppl 40.5 → 373, a
+332 gap — the memory is doing most of the work). The problem is three
compounding, independently-confirmed deficits, none of which requires
abandoning the novel approach:

1. **Catastrophic undertraining** (biggest). 5B tokens = 10 tokens/param.
   SmolLM-360M — a *smaller* model — used 600B (120× more). Coherence has a
   hard token floor and we are ~100× below it.
2. **Zero attention layers.** Every competitive efficient/brain-inspired model
   (MiniMax, Jamba, SpikingBrain) interleaves attention. Pure linear-memory
   models "underperform in retrieval" and on short-context coherence.
3. **Vocabulary too small AND lossy.** 16,384 tokens is ~4× too small for a
   500M model, and the Llama-128k→16k lookup silently *drops ~10% of tokens* —
   deleting content words, which trains the model to emit vague filler.

The fix is to adopt the exact pattern every successful brain-inspired model
already uses (hybrid attention + proper vocab + heavy training + distillation),
while keeping the genuinely novel, differentiated part: the persistent Hebbian
brain state.

---

## Full diagnosis (evidence)

### 1. Undertraining — the dominant factor

- Pretraining used **5.0B tokens** (`logs/500m_v3.txt`: "done: 5000.0M tok"),
  at 479M params = **10 tokens/param**.
- **29.05B tokens are already tokenized on disk** and were never used
  (`/home/glenn/projects/bdh/data/tokenized/train.bin` = 58.1 GB uint16). We
  used ~17% of what we already have.
- Published coherence floors for small models:
  - SmolLM-360M: **600B tokens** (a *smaller* model, 120× our budget).
  - SmolLM2: 2T. TinyLlama-1.1B: **3T tokens**.
  - Even aggressively data-efficient research runs (13B tokens) land "clearly
    behind modern 135M models trained with 46–154× more tokens."
- Final pretrain **val_ppl = 40.5** on the 16k vocab — mediocre; a well-trained
  small model reaches ~10–20 on its native vocab. Consistent with undertraining.

**This is the #1 lever. No SFT dataset, optimizer, or architecture tweak fixes
a base model trained on ~100× too few tokens.**

### 2. No attention layers — a real architectural gap

`src/model.py`: the block output is `out = norm(r + ff)` where `r = M @ q` (a
read from the fixed-size `d_mem × d_mem = 2048²` Hebbian matrix) and `ff` is a
per-token feedforward of the current spike. **There is no softmax attention
anywhere.** Every token's access to prior context is compressed into one
fixed-size matrix per layer (this is the linear-attention / fast-weight class).

Published 2025–2026 findings:
- "Pure Mamba or pure RWKV underperform pure Transformer on short contexts;
  combining state-space layers with attention layers gives the best of both."
- "Linear attention underperforms in retrieval." "Linear RNNs suffer history
  decay beyond training length."
- **MiniMax-01**: Lightning (linear) attention most layers, **full softmax
  attention every 8th layer**. **Jamba**: 1 Transformer layer per 8 Mamba
  layers. **MiniMax-M2 (2026) reverted to full attention in every layer** —
  "no variant reliably matches full attention quality in production."
- **SpikingBrain-7B (TMLR 2026, arXiv:2509.05276)** — the most direct analog to
  this project: **full attention at layers [7,14,21,28], linear + sliding-window
  attention in ALL other layers.** Not one layer is pure-linear-only.

Our model has zero attention of any kind. This is the clearest architectural
gap and the clearest fix.

### 3. Vocabulary — too small and lossily built

- Vocab = 16,384, implemented as a lookup table mapping Llama-3's 128,256
  tokens down to 16,383 slots (`data/vocab_map_16384.npy`), keeping 12.8% of
  source tokens. Everything else maps to 0 and is **silently deleted on encode**
  (`chat.py`: `[lut[t] for t in ids if lut[t] != 0]`).
- Measured token survival on normal English (`temp/diag_vocab.py`): ~90% overall
  but **science/technical text only 75%** — content words vanish (" nephew",
  " ATP", " mitochondria" all → dropped). Function words survive.
- Effect: during pretraining the model literally saw `"The ___ is the ___ of the
  cell"` and learned to fill holes with generic filler; at generation it can
  *never* emit the dropped words because they don't exist in its output space.
  **This is a near-exact mechanistic cause of the "complex and multifaceted"
  vague-filler failure mode.**
- Published: optimal vocab for a 124M model is ~32k and *grows* with scale
  (a 500M model wants 32–64k). "Vocabulary size of 500 consistently performs
  worse… struggles to compose words." 16k is roughly half-optimal for 124M,
  ~4× too small for 500M — before counting the lossy drop.

### 4. Secondary: fixed-threshold spiking + static sparsity penalty

Our LIF neurons use a fixed threshold (vth=1) and are pushed toward 5% firing by
a static penalty (`sparsity_lambda`), fighting the model's natural ~25% rate.
SpikingBrain found this exact regime problematic and switched to
**adaptive-threshold** neurons (dynamic threshold ∝ mean membrane potential) to
"ensure balanced firing rates and prevent over-excitation/quiescence." Our
sparsity penalty is a headwind on representational capacity; worth revisiting,
but lower priority than 1–3.

### What is NOT the problem

- **The Hebbian memory works.** Memory Δ = +332 (ablated ppl 373 vs 40.5). The
  novel mechanism carries most of the model's predictive power. Keep it.
- **The optimizer, MTP, Engram-v2, curriculum, GDN-2 kernel** are all fine —
  individually screened, not the cause of incoherence.
- **The SFT data** was a real weakness (Alpaca-only) and Tulu-3 helps, but SFT
  cannot rescue an undertrained, lossy-vocab base.

---

## The plan — how to get to stellar (prioritized by impact)

The through-line: **stop treating attention as the enemy; adopt the hybrid
brain-inspired pattern that actually works, keep the persistent-Hebbian-memory
differentiator, and train it properly.**

### P1 — Train 20–100× longer (the dominant lever)

- Move from 5B → **at least 100B tokens**, ideally 200B+. Use the 29B already on
  disk, then extend with FineWeb-Edu / DCLM-baseline (high-quality, proven for
  small models).
- Compute reality (home 2-GPU rig): at ~11k tok/s, 100B tokens on a 500M model
  ≈ ~100 GPU-days. This forces one or more of: (a) distillation to get more
  quality per token (P4), (b) both GPUs in parallel, (c) a *smaller* model
  trained *longer* rather than a bigger one trained briefly, (d) accepting a
  multi-week run. **Honest tradeoff: coherence is not free, and "1B trained on
  5–10B tokens" will be worse than "400M trained on 100B tokens."**

### P2 — Add attention (hybrid), keep Hebbian memory as the linear component

- Interleave softmax attention using the proven pattern: **full attention every
  4th–8th layer, sliding-window attention (or the existing Hebbian read) in the
  rest.** This is exactly SpikingBrain's "LA+FA at [7,14,21,28], LA+SWA
  elsewhere."
- The Hebbian/GDN-2 memory becomes the "linear attention" in the hybrid — it
  stays load-bearing and differentiated; we're adding periodic exact attention
  for the short-context coherence and retrieval it can't do alone.
- Smallest viable version to de-risk at 30M/90M: 1 full-attention layer inserted
  into the middle of the stack, measure val_ppl + MQAR recall + qualitative
  coherence vs. the no-attention control.

### P3 — Fix the vocabulary

- Train a proper **32k BPE tokenizer from scratch on the actual corpus**
  (SentencePiece/tokenizers). Every input becomes representable → **zero dropped
  tokens** — the single change that removes the vague-filler failure mode at the
  data level. Re-tokenize the training data once (~hours).
- 32k also roughly doubles context density and matches published optimal for
  this scale. Cost: embedding+head grow from 2×33M to 2×64M params — acceptable,
  and honest (real vocabulary, not a lossy projection).

### P4 — Distill from a strong teacher (the proven spiking path)

- Every successful efficient/spiking model gets quality by **distilling or
  converting from a strong dense model** (SpikingMamba ← Mamba; SpikingBrain,
  LAS, FAS ← dense LLMs via conversion/KD), not by training the exotic
  architecture from scratch.
- Distill from a good open 1–3B teacher (Qwen3-1.7B, Llama-3.2-1B/3B) into our
  hybrid-spiking-Hebbian student. This multiplies quality-per-token, directly
  attacking the P1 compute wall. The project **already has KD infrastructure**
  (`src/kd.py`, used in the 500M v2 line) — re-enable and point it at a current
  teacher.

### P5 — Relax the spike target from 5% to ~15% (biological AND practical)

- The current recipe fights the model's natural firing rate. Pretrain had to
  bump `sparsity_lambda` mid-run from 0.5→1.0 just to hold ~15-20% spike, and
  each bump traded away perplexity and MQAR recall (documented in
  `500m_v3.yaml:85` and `neuro_500m_v3_sparsity_tuning.md`). The penalty is
  a headwind on capacity — dead weight fighting the loss.
- **Set `sparsity_target: 0.15` (up from 0.05)**, and start with a modest
  `sparsity_lambda` (~0.3-0.5). This gives back real representational capacity
  while still delivering an honest **~85% inference-energy reduction vs a dense
  model** — a compelling story. 15% is also more biologically defensible than
  5% (cortical task-active populations fire in the 10-30% range, not 1-5%).
- Screen 15% vs 10% vs 20% at 30M once other changes land; pick the target
  that gives best quality without collapsing the differentiator.

### P6 — Code hygiene: remove legacy branches, keep every proven win

**Not a feature cut — a cleanup.** The 500M v3 recipe already keeps only the
proven wins. The "simplification" is removing DEAD CODE PATHS from earlier
iterations that are permanently False in every current config, but still add
silent-bug surface (we've been bitten repeatedly by `build()` drift across
exactly this sprawl — 5+ recurrences documented).

**KEEP (all proven, all in the current recipe):**
- GDN-2 chunked memory (2-seed 30M win, 90M de-risk, 500M ship)
- Engram-v2 (2-seed 30M win, factual eval confirms)
- NorMuon (confirmed win, 28% slower but real -1.1 ppl)
- MTP (Batch A: 2-seed 30M win, +recall)
- Curriculum warmup (2-seed confirmed)
- MQAR augmentation (throughout the recipe)
- `beta_floor: 0.1` (β-collapse fix — non-negotiable)
- Muon `fix_gate_matrices` + `route_embeddings_by_type` (F1/F2 fixes)
- RMSNorm + embed_scale + z_loss (Batch A pack)
- torch.compile with the LIF-disable fix

**ADD:**
- Hybrid attention (P2)
- 32k tokenizer (P3)
- KD from a strong teacher (P4)
- Relaxed sparsity target 5% → 15% (P5, this section)

**REMOVE FROM CODE (dead branches, not currently used, safe to delete):**
- SDM (CLOSED NEGATIVE, `neuro_sdm_closed.md`)
- OSDN (never made it past early screens)
- Titans (legacy, never in v3)
- Markov head (in v2 legacy, dropped for v3)
- MoR / ponder / adaptive_compute (killed at 350M, deferred to Phase 3)
- Confidence head (v2 legacy, dropped for v3)
- Engram v1 (superseded by v2)

Effect: fewer conditional paths in `model.py`/`train_gpu.py`/`build_model`,
fewer `if X_enabled:` branches for a future drift bug to hide in. Zero
functional change to the committed recipe.

### P7 — (Later) adaptive spiking threshold

- Replace fixed-threshold LIF + static sparsity penalty with SpikingBrain-style
  adaptive-threshold neurons (dynamic threshold ∝ mean |membrane potential|) so
  firing stays balanced without a penalty fighting the model. Revisit after
  P1–P5 land; it's a capacity refinement, not a coherence blocker.

---

## What this means for the "1B" plan

The `1b_v1_design.md` premise — "same de-risked recipe, just bigger" — is
**revised**. Going to 1B without fixing vocab, attention, and training budget
would spend ~a month of GPU to produce a bigger version of the same incoherent
output. The corrected sequence:

1. **Fix the recipe at small scale first** (30M/90M screens): new 32k tokenizer,
   hybrid attention (1 full-attn layer), KD from a strong teacher, simplified
   core. Verify coherence + MQAR recall beat the current no-attention control.
2. **Then pick the scale that the compute budget can actually train to
   coherence.** A 300–500M model trained on 100B+ tokens (with KD) will beat a
   1B model trained on 5–10B. Size follows the token budget, not vice-versa.
3. **Keep the persistent-brain demo** as the differentiator — it is genuinely
   unique and load-bearing, and it's the story no Qwen/Llama can tell.

---

## Pre-registered success criteria (before committing to a full run)

At the 30M/90M de-risk stage, the redesigned recipe must show, vs. the current
no-attention / 16k-vocab control:
- **Qualitative coherence**: on the 13-prompt REPL set (proper sampling), a
  clear majority of prompts get on-topic, non-repetitive, grammatical answers —
  not vague filler or loops.
- **MQAR recall** at N=16/64 at least as good as the current model (attention
  should help, not hurt, recall).
- **val_ppl** measured on the *new* vocab (not comparable across tokenizers) —
  used only for relative screening within the new recipe.
- **No dropped-token failures**: the new tokenizer round-trips arbitrary English
  and code with zero deletions (unit test).

---

## References (ranked)

Primary architectural port targets:
- **SpikingBrain Technical Report (arXiv:2509.05276, TMLR 2026)**,
  `github.com/BICLab/SpikingBrain-7B` — the direct analog: hybrid attention
  (full every 7th layer + linear/SWA elsewhere), conversion/distillation from
  dense, adaptive-threshold spiking. **Port the hybrid-attention layout and the
  conversion-based training idea.**
- **MiniMax-01 / MiniMax-M2 (arXiv:2605.26494)** — hybrid vs. full-attention
  evidence; the 1-in-8 full-attention layout and the finding that full attention
  wins in production.
- **Jamba** — 1 Transformer per 8 Mamba layers; simplest hybrid layout to copy.

Training budget / small-model coherence:
- **SmolLM / SmolLM2** (Hugging Face) — 360M@600B, the concrete coherence-token
  floor for our scale.
- **TinyLlama (arXiv:2401.02385)** — 1.1B@3T.
- "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies"
  (NeurIPS 2024) — optimal vocab ~32k@124M, grows with scale.

Spiking-LLM viability / distillation path:
- **SpikingMamba (arXiv:2510.04595)** — energy-efficient spiking LLM via KD from
  Mamba; evidence that the good path is distill-from-dense.
- **LAS (arXiv:2505.09659), FAS (arXiv:2502.04405)** — loss-less/fast ANN→SNN
  conversion for spiking LLMs; same lesson.
- SpikeGPT (arXiv:2302.13939), SpikeLLM (ICLR 2025) — background on spiking-LM
  difficulty and saliency-based scaling.
