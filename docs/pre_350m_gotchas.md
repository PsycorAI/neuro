# Pre-350M gotcha audit (2026-08-04)

Research + code audit done while the gate-norm arm (G1) runs. Purpose: find
everything that could bite a 350M run **before** we spend GPU-days on it.

Ordered by severity. Items 1-3 I would fix before any 350M launch.

---

## 1. CRITICAL — Muon runs with ZERO weight decay, and *cannot* do otherwise

**What's wrong.** `src/muon.py:58` — the `Muon` class signature is
`__init__(self, params, lr, momentum, nesterov, ns_steps, normuon, ...)`.
There is **no `weight_decay` parameter at all**. Meanwhile
`build_muon_adamw(..., weight_decay=...)` passes the config's weight decay
*only* into AdamW (`muon.py:193`). So:

- The Muon group — `W_k`, `W_v`, `W_q`, `W_ff`, `to_current`, **`W_w`,
  `W_b`**, and the attention `qkv`/`o_proj` — has **no weight decay and no
  way to enable it** without a code change.
- On top of that, `weight_decay` defaults to `0.0` and **no current config
  sets it** (only the legacy `demo_phase1b_500m_v2_fast.yaml` mentions it).
  So even the AdamW group is running undecayed.

**Why it matters — this is likely the ROOT of the Step-3 divergence.**
Moonshot's [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
states plainly:

> "Two crucial techniques for scaling up Muon are: (1) adding weight decay and
> (2) carefully adjusting the per-parameter update scale."
> "Vanilla Muon initially converges faster, but model weights grew too large
> over time, potentially limiting the model's long-term performance."

That is precisely what we measured: weight norms growing **2.0-2.8x** over
12k steps with no bound (`temp/diag_gate_saturation2.py`), followed by a run
that bottomed at step 7000 and then degraded.

The causal chain: no weight decay → weights grow unbounded → gate logit
`W_gate(spk)` grows with them → logit exceeds the sigmoid's useful band →
saturation → gradient ≈ 0 → gates freeze → blocks 1-2 lose their memory →
quality plateaus and degrades.

**Note we already implement their technique #2** — `muon.py:121-122` does
`scale = max(1.0, (max(p.shape)/min(p.shape))**0.5)`. We have half the recipe.

**Fix — LANDED 2026-08-04.** `Muon` now takes `weight_decay` and applies
decoupled decay `p.mul_(1 - lr*wd)` using the unscaled group lr (so decay is
uniform across matrix shapes and automatically tracks the LR schedule);
`build_muon_adamw` passes it through. Defaults to 0.0, so behaviour is
byte-identical unless a config opts in. Verified by `temp/test_muon_wd.py`:
decay is exact to 8.6e-9, reaches the Muon group (was 0.0 before), and bounds
weight growth under a persistent same-direction gradient.

**Still to do: actually set `weight_decay: 0.1` in the 350M config and screen
it.** The code is ready; the hyperparameter is not yet validated on this
architecture. This is upstream of the gate-norm fix — gate-norm bounds the
*symptom*, weight decay bounds the *cause*. Use both.

---

## 2. HIGH — our learning rate is probably too LOW for a full-length run

**Step Law** ([arXiv 2503.04715](https://arxiv.org/html/2503.04715v7)) fits
optimal LR as `η(N,D) = 1.79·N^(-0.713)·D^(0.307)` — note LR rises with
*dataset size*, not just falls with model size.

Plugging in our runs:

| Run | N | D | Step-Law η | We used | ratio |
|---|---|---|---|---|---|
| 90M screens | 107M | 150M | 1.09e-3 | 6.0e-4 | 0.55x |
| Step 3 (as launched) | 276M | 157M | 5.6e-4 | 5.0e-4 | 0.89x |
| Step 3 (full budget) | 276M | 1.1B | 1.02e-3 | 3.5e-4 | **0.34x** |

Two conclusions:

- **The original 5e-4 was NOT unreasonable** — it's ~0.9x Step Law for the
  tokens actually consumed. This corroborates the gate diagnosis: 5e-4
  didn't diverge because it was too hot, it diverged because the gates
  saturated.
- **3.5e-4 is well below optimum for a 1.1B-token run.** Some of our
  quality gap vs the 90M model is probably *under*-training, not just
  breakage. After the gate fix lands, LR should be re-raised — that's what
  arm G2 tests.

**Caveat:** Step Law was fit on dense AdamW transformers, and our `lr` only
governs the AdamW group (Muon has its own `muon_lr`). Treat the *direction*
as the signal, not the absolute number.

---

## 3. HIGH — effective batch may be ~6x too small

Step Law's batch rule `B(D) = 0.58·D^0.571` depends **only on dataset size**.
For D = 1.1B tokens it suggests ~**84k tokens/step**. We run
`batch 28 × block 512 = 14,336 tokens/step` — about **6x smaller**, i.e. ~6x
more optimizer steps each with ~6x noisier gradients.

Noisy updates interact badly with an unbounded-growth gate: noise gets
rectified into logit magnitude. Cheap to test via `grad_accum` (throughput
is nearly unaffected; only the step count changes). Worth one screen before
350M — but lower priority than items 1-2 since it changes the whole LR
schedule shape and would confound the gate test.

---

## 3b. HIGH (FIXED) — energy report benchmarked against a 2-LAYER transformer

`train_gpu.py:822` passed `baseline_layers=c.get("n_layer", 2)`. Spiking
configs set **`n_layers` (plural)** — `n_layer` singular is the
TinyTransformer key — so the lookup **always missed** and every energy report
this project has ever printed compared our 4-8 layer spiking model against a
**2-layer** transformer.

Corrected at matched depth (`temp/recompute_energy.py`):

| config | n/d | printed (L=2 base) | fair (matched L) | @seq 4096 |
|---|---|---|---|---|
| 30M d=384 n=384 L=4 | 1.0 | 1.17x | **1.47x** | 2.23x |
| 90M d=640 n=640 L=6 | 1.0 | 1.12x | **1.90x** | 2.86x |
| 104M d=640 n=1536 L=8 | 2.4 | 0.88x | **1.79x** | 2.79x |
| 104M λ=0.5 | 2.4 | 0.89x | **1.81x** | 2.83x |
| 276M d=1152 n=2752 L=8 | 2.4 | 0.82x | **1.99x** | 2.74x |

**The efficiency claim is ~2x STRONGER than we have been reporting**, and it
*improves* with scale rather than degrading. Every energy figure in past logs,
docs, and any blog material is understated and should be regenerated.

Worth recording honestly: the buggy numbers made it look like the n/d=2.4
aspect ratio had destroyed the energy advantage ("spiking LOSES" at 0.88x),
and I was one step from reporting that as a finding. It was the baseline, not
the architecture. Fixed to `c.get("n_layer", c.get("n_layers", 2))`.

---

## 4. MEDIUM — sparsity penalty has a constant gradient (doesn't scale)

`train_gpu.py:541`: `λ * (sr - target).clamp(min=0)`. The gradient w.r.t.
spike rate is **exactly λ whenever above target, regardless of how far
above**. The CE gradient grows with model capacity; this one doesn't. So a
λ tuned at 107M is systematically too weak at 276M — which is exactly what
we saw (spike ran *away* from target at 276M while falling nicely at 107M).

**Fix:** quadratic hinge `λ * (sr-target).clamp(min=0)**2` → gradient
`2λ(sr-target)`, self-correcting and far more scale-transferable. One-line
change. Recommend doing this *after* the gate arms report, so it doesn't
confound them.

---

## 5. MEDIUM — spike-rate telemetry is diluted by attention layers

`model.py:1259` averages `all_spikes` over **all** layers, and attention
layers contribute exactly zero (`hybrid_attn.py:187`). At L=8 with one
attention layer, reported spike rate = (7/8) × the true spiking-layer rate.
So `sparsity_target: 0.15` really means ~0.171 on the layers that spike.
~14% error today; it grows if we ever add attention layers. Also means the
sparsity *penalty* is applied to a diluted number.

---

## 6. FIXED THIS SESSION — two of three gates were unmonitored

The `[beta]` line only ever reported the **write** gate. The erase and decay
gates were invisible for this entire investigation, which hid:

- block 0's decay-gate logits blown up to **std 7.34** (healthy blocks: 2.2-2.9)
- blocks 4-7 erase gates **58-92% saturated**

Now logged as `[gates sat]` (erase/write/decay saturation + decay-logit std).
**Decay-logit std is the best early tripwire we have** — it separates healthy
from broken blocks more cleanly than anything in the old telemetry.

---

## 7. LOW — dead parameters under gdn2 (diagnostic trap)

`W_beta` / `W_alpha` are still constructed but receive zero gradient when
`decay_gate + gdn2_enabled` (they're superseded by `W_b`/`W_w`/`W_fdecay`).
Their frozen weights cost me ~30 minutes of false "optimizer bug" diagnosis.
Harmless to training; worth skipping their construction for clarity.

---

## 8. LOW — `hybrid_attn` docstring says SwiGLU, config says otherwise

`hybrid_attn.py` documents a "SwiGLU FFN", but the block honors
`use_swiglu`, which is **false** in every Step-3 config — so the attention
block actually runs a plain GELU MLP. Code is correct; the docstring
misleads. If we ever *want* SwiGLU on attention specifically, it needs its
own flag.

---

## Recommended sequence before a 350M run

1. **Land weight decay in Muon** (item 1) — highest expected value, and it
   attacks the root cause rather than the symptom.
2. Let **G1 (gate-norm)** report. Gate-norm and weight decay are
   complementary: WD bounds weight growth, gate-norm bounds the logit
   regardless of weight growth. Belt and braces for a multi-day run.
3. **G2: re-raise LR** (item 2) once gates are stable — this is where the
   quality upside is, and where we close the embarrassing gap to the 90M model.
4. Then quadratic sparsity (item 4) and the batch-size screen (item 3) as
   separate, non-confounded screens.
5. Only then commit to 350M.

## Honest note on what we still don't know

Neither SpikingBrain (arXiv 2509.05276 — the closest published
spiking+hybrid-attention model) nor the Linear Attention Architectures
survey documents gate-collapse-at-scale. Either it's specific to our
combination (spiking LIF front-end feeding GDN-2 gates — nobody else has
that exact stack), or others hit it and solved it silently via weight decay,
which everyone else uses by default and we do not. I lean toward the latter,
which is why item 1 is ranked first.
