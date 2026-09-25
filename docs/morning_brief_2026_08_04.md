# Morning brief — 2026-08-04

## Headline: Step 3 diverged. Killed it, diagnosed it, fixes are running.

Full detail: memory `neuro_step3_divergence_2026_08_04.md`.

## What happened

Step 3 (276M, λ=0.5, lr=5e-4, 1.1B-token target) launched fine, then:

- **val_ppl bottomed at 97.93 (step 7000), then climbed monotonically** to
  115.28 by step 17000.
- **spike rate CLIMBED 0.325 → 0.369** *against* the sparsity penalty. At
  90M with the same λ=0.5 it FELL (0.280 → 0.160).
- **Write gates in blocks 1-2 collapsed to the floor** (at_floor 0% → 72%/75%,
  accelerating); blocks 4-7 pinned high (μ 0.90-0.96).

Killed at step ~17000. The `&&` chain short-circuited, so seeds 1 and 2 never
launched — no wasted GPU time beyond seed 0.

## Diagnosis

Checkpoint forensics (`temp/diag_gate_saturation2.py`) showed the anomaly is
**gate-specific, not a global weight blowup**:

| tensor | blk1 | blk2 | other blocks |
|---|---|---|---|
| W_fdecay | 3.09x | 3.92x | 1.36-1.68x |
| W_b | 2.75x | 2.48x | 1.38-2.06x |
| W_k / W_v / to_current | 2.1-2.5x | 2.2-2.5x | 1.9-2.2x (uniform) |

**Leading hypothesis: λ=0.5 is too weak at 276M.** The penalty is
`λ * (spike_rate − target).clamp(min=0)`, so its gradient w.r.t. spike rate
is a *constant* λ regardless of scale, while the CE gradient grows with model
capacity. Fixed λ gets relatively weaker as the model grows → spike runs away
→ gate pre-activations saturate the sigmoid → write gates pin at floor →
memory pathway breaks → divergence.

Direct precedent: **500M v3 needed λ bumped 0.7→1.0 mid-run** for exactly
"spike still climbing rapidly at 0.24+". Literature agrees that saturated
sigmoid gates suffer vanishing gradients and cannot recover on their own
(ReGLA arXiv 2502.01578, Gated Attention arXiv 2505.06708).

Secondary hypothesis: lr=5e-4 (which I log-interpolated and never tested at
d=1152) is too hot.

## My mistake, stated plainly

I carried λ=0.5 from a 107M screen to a 276M run. The project already has a
memory warning about exactly this class of error
(`neuro_sparsity_refine_ratio_gotcha`: match λ as a *ratio*, never copy an
absolute value across differently-sized configs). The 30M/90M screen ladder
could not have caught it, because λ's *relative* strength is scale-dependent
— but I should have re-tuned rather than inherited.

## Running now (5090, ~3.5h each, ~10.5h total)

Three diagnostic arms at the true Step 3 shape, baseline = the killed run's
own logged trajectory:

| arm | λ | lr | tests |
|---|---|---|---|
| `step3diag_lam1` | 1.0 | 5e-4 | isolates λ (highest prior) |
| `step3diag_lr35` | 0.5 | 3.5e-4 | isolates lr |
| `step3diag_lam1_lr35` | 1.0 | 3.5e-4 | combined |

**Decision criteria at ~step 8000** (killed baseline: val_ppl ~101-103 and
turned up, spike 0.339, blk1 at_floor 38-50%):
- healthy = val_ppl still falling, spike falling, at_floor ~0%
- broken = val_ppl turned up, spike climbing, at_floor >20%

## A trap I hit and fixed (worth knowing)

My first version of these arms set `max_tokens=120M` to make them cheap. But
`train_gpu.py` derives **both** LR warmup and the cosine-decay denominator
from `total_steps = max_tokens // per_step_tok` — so those arms had
`warmup=167 | total_steps=8370` and would have decayed LR to near-zero right
where the real run was still at peak. They'd have looked healthy and proven
nothing. Caught it from the banner line ~2 minutes in, killed, and relaunched
with `max_tokens` kept at 1.1B and **wall-clock `timeout`** doing the
truncation instead. Recorded in memory for future truncated diagnostics.

## State

- **5090**: running the 3 diagnostic arms (chained with `;` so a timeout
  advances rather than aborts)
- **5080**: had ~10 GB in use by your app — left alone
- Step 3 seed 0 checkpoints retained at `models/checkpoints/step3_267m_s0/`
  (steps 4000/8000/12000/16000) for any further forensics
- `configs/step3_267m_s0.yaml` is **not safe to relaunch as-is** — needs the
  λ (and possibly lr) fix from these arms first

## What I'd propose once the arms report

1. Take the arm that's clearly healthy at step ~8000-9000 and re-run Step 3
   seed 0 with it, full 1.1B tokens.
2. Add a standing **early tripwire**: spike-rate *direction* in the first
   ~5000 steps. It turned up ~2000 steps before val_ppl did — cheapest
   possible early warning for the remaining seeds and for Step 4.
3. If *no* arm is healthy, the next suspects are (a) `attention_heads=9`
   (the one genuinely untested thing I introduced — 8 heads/head_dim 144 is
   the safe alternative), and (b) a real fix to gate saturation rather than
   a hyperparameter workaround.
