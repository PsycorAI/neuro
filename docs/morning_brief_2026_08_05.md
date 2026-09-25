# Morning brief — 2026-08-05

Overnight work while you slept: two stability fixes implemented + unit-tested,
a separate bug found + fixed, a clean config audit, and a 3-arm validation
matrix running on both GPUs. Details below; live results table at the bottom.

---

## The story so far (recap)

Step 3 (276M) diverged. Root cause, now well-established:

- **No weight decay anywhere.** The `Muon` class had no `weight_decay`
  parameter at all, and no config set one. So `W_w`/`W_b`/`W_k`/`W_v`/
  `to_current` + attention matrices grew unbounded (measured 2.0-2.8x over
  12k steps). Moonshot's Muon paper names this exact failure and lists weight
  decay as a required technique for scaling Muon.
- Unbounded weights → GDN-2 gate logits grow past the sigmoid's useful band →
  gates saturate → gradient dies → blocks 1-2 lose their memory → divergence.
- It bit at 276M but not 107M because more active spike units = larger
  logits = crosses the saturation threshold sooner.

## What I implemented (code landed, unit-tested, back-compat preserved)

1. **Muon weight decay** (`weight_decay`, default 0.0) — the root-cause fix.
   Decoupled `p*=(1-lr*wd)`, applied to BOTH optimizers. The AdamW group is
   now split so decay hits only 2D weights, never RMSNorm gains / biases /
   `gate_bias_f` / `gdn2_a` (verified: 12 decay / 44 no-decay params; decay
   exact to 8.6e-9). `temp/test_muon_wd.py`, `temp/test_wd_grouping.py`.

2. **Gate-norm** (`gdn2_gate_norm`, default false) — the symptom fix. RMSNorm
   on the three GDN-2 gate pre-activations, with a `gate_bias_f` param that
   preserves W_fdecay's -3.0 slow-decay init. `temp/test_gate_norm.py` (5
   checks incl. real 276M ckpt loads clean, compile works). Partial
   validation earlier (arm G1): clean through step 4500, but killed before
   the danger zone.

Both default OFF → every prior checkpoint/run is unaffected.

## Separate bug found + fixed: energy report used a 2-layer baseline

`train_gpu.py:822` read `c.get("n_layer", 2)` but configs write `n_layers`
(plural). The lookup always missed, so every energy report compared our 4-8
layer model to a **2-layer** transformer. Fixed. Matched-depth recompute:

| model | printed ratio | **correct ratio** | @seq 4096 |
|---|---|---|---|
| 90M | 1.12x | **1.90x** | 2.86x |
| 276M | 0.82x ("loses") | **1.99x** | 2.74x |

**The efficiency advantage is ~2x stronger than we've been claiming and
GROWS with scale.** All past energy numbers (logs, docs, blog) are understated
— regenerate them. A full config-key audit (`temp/audit_config_keys.py`)
found no other silent-key bugs of this class; `grad_clip` (1.0) and
`eval_iters` (100) are both real and applied.

## Overnight validation matrix (running now)

All at the diverged 276M shape, schedule-faithful (max_tokens=1.1B, truncated
by wall-clock). Baselines: original Step-3 (val_ppl bottomed 97.93 @ step 7000
then diverged) and arm 3 (gn=0/wd=0/λ=1.0/lr=3.5e-4, blk1 at_floor→66%).

| arm | GPU | gate-norm | wd | lr | λ | what it answers |
|---|---|---|---|---|---|---|
| `step3fix_gnwd` (A) | 5090 | on | 0.1 | 3.5e-4 | 1.0 | conservative recipe — does the combined fix hold? |
| `step3fix_gnwd_hot` (B) | 5090 | on | 0.1 | 5e-4 | 1.0 | can we run at Step-Law-preferred higher lr for quality? |
| `step3fix_gn_only` (C) | 5080 | on | 0 | 3.5e-4 | 1.0 | is weight decay needed on top of gate-norm? |

Success = blk1/blk2 `at_floor` stays low (<~10%) through the step 5500-9000
danger zone AND val_ppl keeps descending (arm 3 bottomed ~62 then flattened).

**Hypothesis to check in the data:** weight decay on `to_current` may also fix
the spike runaway (0.325→0.369), since bigger `to_current` → more LIF current
→ more spikes AND bigger gate logits, one shared cause. If arm A's spike rate
stays flat with no sparsity-penalty change, the quadratic-penalty idea (gotcha
#4) is unnecessary — don't add it pre-emptively.

## Next-scale shape options (ready; pick after the fix confirms)

`temp/sweep_350m_shapes.py`, at the confirmed aspect ratio (n≈2.4d), 1
attention layer, head_dim 128:

| shape | params |
|---|---|
| d=1152 n=2752 L=8 (current) | 276M |
| d=1280 n=3072 L=8 | 332M |
| **d=1280 n=3072 L=9** | **363M** ← cleanest ~350M |
| d=1344 n=3200 L=8 | 359M (head_dim 168, less clean) |

Open question for you: after the fix confirms, do we (a) re-run Step 3 at 276M
3-seed to finish the *original* mid-scale seed-variance goal, then scale, or
(b) jump straight to ~350M? I lean (a) — it's the cheaper way to confirm the
fix is seed-robust, and it completes the gate Step 3 was always meant to be.

## Full gotcha list

`docs/pre_350m_gotchas.md` — 8 items, ranked, with status. Items 1 (weight
decay) and 3b (energy bug) now fixed; gate-norm covers the symptom; remaining
open items (quadratic sparsity, batch size, spike telemetry dilution) are
lower priority and may be moot depending on tonight's results.

---

## LIVE RESULTS + an incident

### Incident: arm A (gn+wd conservative) OOM-KILLED at step 2500

Host RAM, not GPU. `free -g`: 23 GB total / 0 free. Running two 276M compiled
runs simultaneously (arm A on 5090 + arm C compiling on 5080) spiked host RAM
past the WSL 24 GB cap and the OS OOM-killer SIGKILLed arm A — clean process
disappearance, no Python traceback, throughput collapsed 9,000→1,150 tok/s
just before. This is the known `.wslconfig` cap issue
([[reference-wsl-memory-config]]).

**Operational lesson: never run two 276M-class compiled runs at once under the
24 GB WSL cap — the compile-time RAM peak (2-3x steady state) blows the cap.**
Serialize heavy runs, or raise the `.wslconfig` memory cap before parallel
runs. The double-compile window is the danger; steady-state coexistence (17 GB
available) is fine, which is why arm B + arm C continue to run.

Through step 2500 arm A was healthy and tracking G1 exactly (val_ppl 102.72,
blk1/2 at_floor 0%). Lost before the danger zone — but **arm B covers gn+wd at
the more-informative hot lr**, so the loss is tolerable.

### Danger zone = steps ~5500-9000 (5090 arms) / equivalent tokens on 5080.

Arm trajectories (val_ppl / blk1 at_floor), baseline = arm 3 which bottomed
97.93 then diverged, blk1 at_floor → 66%:

**Update: serialized to one run.** Arm C's throughput collapsed 4,500→870
tok/s once arm B ramped up (host-resource contention, the same pressure that
killed arm A). Rather than risk a second OOM-kill, I killed arm C (the
secondary wd-ablation, on the slow 5080) and let **arm B run alone on the
5090** — clean, fast, zero contention (RAM now 20 GB available). Arm C was
healthy to step 3000 (val_ppl 105, gates clean); can be re-run alone later if
needed.

**BOTH arm A and arm B then died at ~step 2500 — root cause corrected.**
It was NOT the host OOM-killer and NOT weight decay. dmesg shows
`dxgk: dxgkio_make_resident: Ioctl failed: -12` (ENOMEM) ×3 — the WSL GPU
memory-residency layer failing to fit an allocation, i.e. a **GPU VRAM OOM**.

Cause: **gate-norm's extra activation memory (3 RMSNorms × d_mem × 7 spiking
blocks) + batch 28 exceeded the 5090's dedicated VRAM.** It spilled into WDDM
shared memory (throughput 9k→1.1k tok/s = the paging thrash) and then a
residency allocation hard-failed at ~step 2500 as the caching allocator
fragmented. Arm C (batch 12) survived because lower batch = less activation
memory. Weight decay was confounded with batch size across the arms but adds
no VRAM — the residency error confirms memory, not numerics.

**Fix: gate-norm runs need a smaller batch.** Relaunched arm B ALONE at
batch 16 / grad_accum 2 (eff batch 32, ~57% of the batch-28 activation peak).
The 350M config must budget VRAM for gate-norm's activations, not inherit
arm 3's batch 28.

Arm B (gn+wd, hot lr 5e-4), batch 16/ga2, alone — through the zone:

| step | val_ppl | blk1 at_floor | note |
|---|---|---|---|
| (filling in as evals arrive; must first clear step 2500 where A/B died) | | | |
