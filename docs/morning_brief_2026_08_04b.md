# Morning brief — 2026-08-04, overnight update

Follows `morning_brief_2026_08_04.md` (the divergence + first-plan writeup).
This one adds the audit, the literature research you asked for, and one new
5080 experiment I set going.

---

## 1. Code / config audit findings

Reviewed the code paths recent Step 3 work touched. Nothing else diverging-bad;
a few real observations worth calling out.

### 1a. Sparsity penalty is a **hinge loss** with a **constant gradient**

`train_gpu.py:541`:
```python
main_loss = main_loss + c["sparsity_lambda"] * (sr - target).clamp(min=0)
```

- Gradient w.r.t. `sr` is **either 0 or exactly λ**, regardless of how far
  above target the spike rate has drifted.
- CE gradient grows with model capacity; **the sparsity signal doesn't**. So
  a λ that worked at 107M is *relatively weaker* at 276M — exactly the
  divergence direction we saw.
- **Cleaner fix than tuning λ per scale**: switch to a quadratic
  `λ * (sr - target).clamp(min=0)**2`, which gives gradient
  `2λ * (sr - target)` — self-correcting pressure that scales with the
  violation. Or a relative form `λ * (sr/target - 1).clamp(min=0)`. Both
  would make λ substantially more scale-transferable.
- This is code change, not just a config tweak — worth a small A/B when
  time permits, but NOT blocking Step 3. Note for the recipe redesign
  after Step 3 succeeds.

### 1b. Spike-rate telemetry is **diluted** by the attention layer

`model.py:1259`:
```python
spike_rate = torch.stack([s.mean() for s in all_spikes]).mean()
```
Attention layer `all_spikes[3]` = zeros (documented in
`hybrid_attn.py:187`). At L=8 with attention at [3], reported spike_rate
= (7/8) × true spiking-layer rate. So `sparsity_target: 0.15` in config
effectively means "spiking layers must sit at 0.15 × 8/7 ≈ 0.171." Small
effect (~14% high), documented but not corrected. If we add more attention
layers later this gets bigger.

### 1c. GDN-2 gate init is asymmetric

`model.py:255-259`: `W_fdecay` has **carefully engineered init**
(`weight *= 0.01`, `bias = -3.0`) to bias the decay gate toward "slow
decay" at init. **`W_w` (write) and `W_b` (erase) have NO custom init** —
just PyTorch's default Kaiming uniform. That means at init they produce
sigmoid outputs randomly near 0.5, which is fine — but there's nothing
holding them near mid-range as training progresses. Under the current
recipe's dynamics we saw them drift to the saturated tails (blk1/2 at_floor
72%/75% at step 15000). Directly analogous to Griffin's forget-gate=0.999
explicit init and Kimi's mid-range regularization (see §2).

### 1d. Not a bug, but a maintenance risk

`W_beta` and `W_alpha` are still instantiated and receive zero gradients
under `decay_gate + gdn2_enabled` (they're replaced by W_w/W_b/W_fdecay).
Not causing problems but future me (or you) will spend another 30 minutes
diagnosing the "frozen weights" red herring if we look at gate behavior.
Two-line fix: skip creating W_beta/W_alpha when they're going to be dead.
Optional cleanup, not urgent.

### 1e. Not a confound but worth naming

Under `use_swiglu: false` (Step 3's actual config), the **attention block's
FFN is GELU-plain**, not SwiGLU, despite `hybrid_attn.py`'s docstring saying
"SwiGLU FFN." The block honors the config flag correctly; only the docstring
is misleading. If we ever want the SwiGLU form on attention specifically,
we'd need to force it in the model.py call (or expose a separate
`attention_use_swiglu` config).

---

## 2. Literature research (June–August 2026)

Most-relevant findings for our current issue:

### 2a. Kimi Linear (Moonshot AI, arxiv 2510.26692 with 2026-06-07 preprint)

Ships 48B/3B-active MoE, 3:1 KDA:MLA hybrid. Two things map directly to us:

- **Same 3:1 attention ratio as Qwen3-Next** — consistent with the field's
  convergence, but per our Phase 1 finding this ratio broke recall in our
  recipe (their "MLA" is quite different from our full softmax attention).
- **They explicitly regularize gates during pretraining to prevent
  collapse — "biasing toward mid-range decay values early in training."**
  This is exactly the failure mode we hit at 276M. Their fix would map onto
  our gates as: add an auxiliary loss penalizing gate values far from 0.5,
  or initialize gate biases toward 0 with reduced weight scale.

### 2b. Linear Attention Architectures study (arxiv 2607.07953, 2026-07)

Directly tested our exact stack (GDN family + Muon+AdamW) at 350M-hybrid
scale for 15B tokens. **Very relevant finding**:

- At 350M hybrid, **Muon prefers lr ≈ 3e-4** (and 1e-4 is close). AdamW
  variants prefer 1e-3. **Our Step 3 used muon_lr=5e-3** — 16× higher than
  their empirical optimum.
- Our muon_lr=5e-3 was locked at 500M v3 (all-spiking) and never re-tuned
  when the hybrid attention layer was added at 90M+.
- Their WSD schedule differs from our cosine, so absolute LR values don't
  transfer directly, but the general direction — **muon_lr too hot on
  hybrid stacks** — is a real hypothesis worth testing.

### 2c. DeepSeek-V4 (2026-06)

Compressed Sparse Attention + Heavily Compressed Attention (CSA + HCA).
Bigger architectural change than we should be doing right now — flag for
future roadmap, not for Step 3 recovery.

### 2d. MiniMax M2.5 (2026)

**Went the opposite direction — dropped their previous hybrid design and
returned to plain multi-head attention with QK-RMSNorm** for "reliability
and training stability." Their reason: they couldn't find a hybrid variant
that reliably matched full-attention quality in production. Not
recommending we follow (we've committed to hybrid for the memory
differentiator), but worth knowing that the field's biggest hybrid
skeptic ships plain MHA + QK-RMSNorm — exactly what our attention block
already uses.

### 2e. Comba (arxiv 2506.02475, June 2025)

"Closed-loop control" refinement of Gated DeltaNet. 40% throughput win vs
GDN, competitive quality. Interesting for a future architecture iteration
but not for Step 3 recovery.

---

## 3. What I launched on the 5080 (complementary to the 5090 arms)

The 5090 arms test λ and AdamW-lr. Neither tests **muon_lr**, which the
Linear Attention Architectures paper flags as scale-sensitive. So I set up
one control on the 5080:

`configs/step3diag_muon3e3.yaml`: byte-identical to the killed Step 3
config (λ=0.5, AdamW-lr=5e-4, same schedule/shape/etc.) with **only
muon_lr dropped 5e-3 → 3e-3**. Batch reduced 28 → 14 with grad_accum 2 to
fit the 5080's 16 GB (effective batch preserved at 28). Wall-clock bound
32400s (~9h).

Same evaluation logic as the 5090 arms: healthy = spike falling, val_ppl
not turning up, blk1/2 at_floor ~0% at step ~4000-5000.

---

## 4. Standing lessons captured

Added to memory:
- `neuro_step3_divergence_2026_08_04`: full divergence diagnosis + arm
  design + the schedule-fidelity trap + the W_beta red-herring warning.
- Standing rule already there: `neuro_sparsity_refine_ratio_gotcha` (match
  λ as a ratio, never copy across configs). I didn't apply my own rule.

For the next scale-up beyond this one:
- The three most important checks BEFORE a multi-day run:
  1. **Re-tune λ at the target scale** with a 200-500M-token screen (short
     enough to be cheap, long enough to see if spike-rate turns).
  2. **Re-tune muon_lr at the target scale** — same reasoning.
  3. **Spike-rate direction in first ~5000 steps** is the cheapest early
     tripwire (turned up ~2000 steps before val_ppl did).

---

## 5. Suggested next steps (for your call)

1. Wait for 5090 arms and 5080 muon-lr control to report. Together they
   cover the λ / AdamW-lr / muon-lr axes.
2. **If any arm shows healthy trajectory** (spike falling, val_ppl still
   declining at step ~5000), relaunch Step 3 with that config for the full
   1.1B tokens across 3 seeds.
3. **If none look healthy**, the next candidate fixes in order of my
   prior:
   - **Init W_w and W_b with a mid-range bias** (weight *= 0.1, bias = 0)
     — mirrors Kimi's regularization and W_fdecay's own careful init.
     Code change, ~5 lines.
   - **Quadratic sparsity penalty** (§1a) — code change, one line.
   - **Add explicit gate-collapse penalty** to the loss (KDA-style) —
     `μ * mean(abs(sigmoid(W_w_out) - 0.5) - 0.4).clamp(min=0)` or an
     entropy-maximization term. Cleanest fix but needs care.
   - **Raise beta_floor 0.1 → 0.15 or 0.2** — softer clamp, more room
     before saturation is fatal.
4. **Any relaunch should add a spike-rate-direction tripwire** to the eval
   loop — auto-kill if spike rises for 3 consecutive evals. Cheap
   insurance against another wasted 2-day run.

---

## 6. Open questions I'd want your input on

- **How committed are we to λ = 0.5 / target = 0.15?** The 500M v3 recipe
  ended at λ = 1.0 / target = 0.05. Higher target = more allowed spiking =
  easier to satisfy, less pressure to collapse. Would you prefer
  λ = 1.0 / target = 0.15 (what arm 1 tests) or λ = 0.5 / target = 0.25
  (softer target, unchanged λ)? Both should prevent the collapse; the
  quality/sparsity trade-off differs.
- **Fix or bypass the fixed-λ hinge loss?** Quadratic form is a small but
  real architecture change — do we lock it in now, or bank the finding for
  post-Step-3?
