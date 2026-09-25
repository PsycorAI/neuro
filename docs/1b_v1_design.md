# 1B v1 — next-scale pretrain (design, no code yet)

Date: 2026-07-29. Status: **scoped, DO NOT LAUNCH YET** — gated behind three
pre-launch checks below (500M-v3 demo verdict, 90M same-recipe reconfirm,
parameter-count + VRAM probe on the true 1B shape).

Successor to `500m_v3.yaml`. Same de-risked recipe, wider/deeper. Target
audience: community + investor demo — so quality (both benchmark + qualitative
chat) matters more than pushing new architecture bets. No new architectural
pieces are proposed in this doc; the design leans on 500M v3's already-de-risked
stack. Any *new* piece added on top must go through its own 30M/90M gate
first (see [Not doing (deferred additions)](#not-doing-deferred-additions)).

---

## 1. What it is

A ~1.0B-param pretrain using the exact 500M v3 recipe (GDN-2 chunked +
Engram-v2 + NorMuon + curriculum + MQAR-aug + MTP + sparsity + torch.compile),
scaled up in depth and width. Same tokenizer (vocab=16384), same data source
(`/home/glenn/projects/bdh/data/tokenized/train.bin`), same post-pretrain
pipeline shape (sparsity refine → SFT; DPO **dropped or heavily gated** per
[[neuro-500m-v3-final-verdict]]).

Explicitly **not**:
- A new architecture research effort. No SDM revival, no MoE, no per-token
  MoR router, no attention layers, no DSA-style sparse attention. The 500M v3
  arch is settled; this run is a scale test of the settled arch.
- Chinchilla-optimal. 1B Chinchilla-optimal would be ~20B tokens (~40 days on
  the 5090); target here is 10–15B tokens (10–15 tok/param), matching the
  500M v3's own ~10.4 tok/param ratio.
- A superset of features. We are NOT folding in KD, Markov, or confidence
  head from the `demo_phase1b_500m_v2_fast` legacy config — none of those
  survived the v3 recipe screening (see 500m_v3.yaml header comment lines
  47–53); adding them now would confound the scale test.

---

## 2. Why now vs. why defer

**In favor of building now:**
- The 500M v3 recipe is fully de-risked at 30M and 90M, and shipped at 500M
  without recipe surprises (single mid-run sparsity λ bump was the only tweak).
  This is the least-risky moment we will have to test whether the recipe
  generalizes to 1B before adding new architecture on top.
- Community/investor demo audience needs a bigger, more capable model than
  500M v3 to be competitive with published community 1B releases (Qwen3-0.6B/
  1B, Instella 3B, small open-weight releases). The 500M v3's val_ppl 43.3
  with 89/47/22% recall is respectable at 500M but ordinary at 1B-open-weight
  standards.
- Aligns with 2026 SOTA: MiniMax M2 series uses Gated DeltaNet family;
  Qwen 3.5 Next has GDN layers. Our GDN-2 chunked kernel already fits this
  design pattern.
- Post-500M-v3 architectural backlog is empty of *promoted* items. The last
  three architectural bets (SDM CLOSED-NEGATIVE, Fracconn CLOSED-NEGATIVE,
  uniform MoR CLOSED-NEGATIVE) all failed the 30M/90M gates; the only
  post-500M idea in flight (per-token MoR router) is blocked on a
  never-built sparse-gather kernel and deferred to Phase 3. **Doing more
  arch research before a scale-up would be spinning wheels.**

**In favor of deferring / doing something else first:**
- **500M v3 demo readiness is UNVERIFIED.** The handoff explicitly lists the
  REPL demo prompt curation as the first task of the next session. If 500M v3
  is *not* demo-worthy for reasons **1B won't fix** (SFT template
  mismatches, brain-mod immaturity, tokenizer artifacts, chat pipeline bugs),
  spending 30–40 days on a 1B pretrain that ships with the same problems is
  pure waste. Verify this first (see gate #1 below).
- The 1B run will consume the 5090 for ~30 days. That kills every other 30M/
  90M screen, brain-mod bake, and post-pretrain experiment for a month.
  Nothing else runs in parallel on the 5080 either (post-pretrain stages need
  a base to work on and would just be waiting).
- No new arch bet is being tested. If the 1B is "500M v3 but bigger,"
  producing it is a scaling exercise, not a research result. That is a fine
  thing to do for a demo, but the doc should be honest about it.

**Conclusion:** proceed to design-doc-ready state (this file), but **DO NOT
LAUNCH** until the pre-launch gates in [§10](#10-recommendation--decision-gate)
clear.

---

## 3. Where it plugs into the actual codebase

The 1B is a byte-additive change: one new pretrain config, downstream configs
patched to reference the new base. No src/ changes required if parameter count
lands on the target and the recipe is unchanged.

**New files (proposed):**
- `configs/1b_v1.yaml` — pretrain (structurally identical to `500m_v3.yaml`,
  with the sizing table in [§5](#5-architecture--sizing) applied).
- `configs/sparsity_refine_1b_v1.yaml` — first refine pass (ratio-matched
  to whatever pretrain λ ends up; see [§8](#8-post-pretrain-pipeline)).
- `configs/sparsity_cleanup_1b_v1.yaml` — post-SFT cleanup (6× ratio, same
  as `sparsity_cleanup_500m_v3.yaml:65`).
- `configs/sft_1b_v1.yaml` — full-FT SFT (per
  [[neuro-lora-at-30m-plateau]]; LoRA is post-hoc if we want swap-mods).

**Existing entry points (no changes needed):**
- `src/train_gpu.py:89` — `build_model(c)` already consumes `d`,
  `n_neurons`, `d_mem`, `n_layers` from config. Confirmed via 500M v3.
- `src/train_gpu.py:323` — `bin_path = c.get("data_bin", BIN)` (BIN at line
  22 = `/home/glenn/projects/bdh/data/tokenized/train.bin`). Same source as
  500M v3.
- `src/train_sft.py` — delegates model build to `train_gpu.build_model` per
  the drift-fix ([[neuro-train-sft-build-drift-fix]]). Safe.
- `src/train_dpo.py` — same delegation. If DPO is used at 1B (heavily
  gated), the alignment stage `configs/dpo_1b_v1.yaml` mirrors
  `dpo_500m_standard_v3.yaml` with early-stop via `gen_check()` (from
  §46 of journey.html).
- `scripts/recall_eval.py`, `scripts/chat.py`, `scripts/lm_eval.py`,
  `scripts/engram_factual_eval.py` — all delegate to `train_gpu.build_model`
  after the [[neuro-train-sft-build-drift-fix]] pass. No script changes.

**Constraints already baked into the code that this design must respect:**
- `torch.compile` at 500M/block_size=640 took ~13 min to trace (journey.html
  §45 "third scare"). At 1B/block_size=640 expect **20–30 min compile
  time**. Launch script must not treat 20 min of 0% GPU util as a hang.
- `gdn2_chunked_dense` in `src/delta_chunked.py` is memory-loop-checkpointed
  at chunk=32 ([[neuro-gdn2-dense-chunked]]). At 1B, per-timestep working
  set doubles; chunk=32 should still fit, but VRAM probe is required.
- `snapshot_brain: false` is standing recipe from journey.html §45
  pre-launch audit; carry forward.
- MQAR-aug is separately compiled at `mqar_N=16` fixed (500m_v3.yaml:82).
  Same pattern for 1B; do **not** re-open dynamic-N without another compile
  screen.
- `beta_floor: 0.1` is required for delta_rule to avoid gate collapse
  ([[neuro-beta-collapse-fix]]).
- `muon_fix_gate_matrices: true` + `muon_route_embeddings_by_type: true`
  (F1+F2, both confirmed at 30M+90M+500M).

---

## 4. Cost & risk

### Compute cost (5090-only pretrain)

| Metric | 500M v3 (measured) | 1B v1 (projected) | Basis |
|---|---|---|---|
| Params | 479M | ~1.0B | design target |
| Tokens (pretrain) | 5.0B | 10–15B | 10–15 tok/param (same ratio as v3) |
| Throughput (tok/s) | ~11,343 | ~5,500 | roughly 1/2 of v3 (2× FLOPs/step) |
| Wall-clock (pretrain) | ~5 days | ~20–30 days | tokens / tok/s |
| Wall-clock (SFT, 500M tok) | ~4h | ~8h | 2× FLOPs |
| Wall-clock (sparsity refine + cleanup) | ~9h | ~18h | ~150M + 265M tok, 2× FLOPs |
| **Total pipeline wall-clock** | ~6 days | **~22–32 days** | pretrain + all post-pretrain |

The tok/s projection is a rough halving; will refine after VRAM probe at 1B
shape shows what `batch_size`/`grad_accum` combination actually fits and what
compile mode is stable. If tok/s comes in worse than 5k, budget slips
proportionally.

### VRAM cost (5090, 32 GiB dedicated / 31.8 GiB effective)

500M v3 sweet spot: `batch_size: 14, grad_accum: 2, block_size: 640` →
~29 GiB dedicated ([[neuro-500m-v3-vram-sweetspot]], 500m_v3.yaml:71).

1B v1 projected: activations scale roughly linearly with param count, so
naively at same `batch_size: 14` we'd overflow. Two candidate settings:

- **Option A**: `batch_size: 8, grad_accum: 4, block_size: 640` — effective
  batch 32 (was 28), similar throughput per effective-step, VRAM headroom.
- **Option B**: `batch_size: 6, grad_accum: 5, block_size: 640` — safer
  margin, slightly slower.

**Actual choice must come from a live VRAM probe on the 1B shape** — same
methodology as 500m_v3.yaml:71 (never trust a synthetic estimate at this
scale, per journey.html §45 "second scare" where the batch=16 estimate
overflowed live).

### Risks

- **Recipe silently doesn't scale.** The recipe was validated at 30M/90M/500M
  but not beyond. At 1B, sparsity may need different λ (elasticity curve
  shifts with model size; screen at 90M with the new size-adjusted λ before
  committing). Memory-pathway may need different Engram-v2 K (currently 8192;
  scaling to 16384 doubles the bucket count and is a real change worth
  screening).
- **Confound with 500M v3 verdict.** If we launch 1B *before* running the
  demo-prompt curation on 500M v3, and 1B disappoints, we won't know if the
  problem is scale or the recipe itself. This is the strongest argument for
  the demo-verify gate below.
- **Multi-week unattended run.** The 500M v3 run survived one Windows reboot,
  one WDDM VRAM spill, and one WSL auto-upgrade kill (three separate incidents
  in ~5 days). At 30 days, expect ~6× more of these. Auto-resume paths were
  hardened during the 500M v3 run — but a 30-day run needs the run script to
  be checked against every known kill mode
  ([[reference-wsl-auto-upgrade-kill]], [[reference-wsl-memory-config]],
  journey.html §45 all three scares).
- **DPO stage lands nothing** at 500M and it's unclear whether that's a
  scale artifact or a fundamental limitation of the recipe with a 60k-pair
  preference set. At 1B, the same failure mode is *more* likely (overfitting
  on repeat passes is worse at higher capacity), not less. **Design decision:
  skip DPO at 1B unless a 30M/90M DPO screen with `gen_check()` early-stop
  shows a real behavioral gain over SFT-only.**
- **Wasted GPU time if 500M v3 wasn't demo-worthy for a reason 1B won't
  fix.** Gate #1 below addresses this.

---

## 5. Architecture & sizing

**Target: ~1.0B params** (exact number set by parameter-count dry-run before
launch — must be ≥ 900M and ≤ 1.1B to count as "1B").

**Proposed shape (starting point; verify with build_model dry-run):**

| Field | 500M v3 | 1B v1 (proposed) | Notes |
|---|---|---|---|
| `vocab` | 16384 | 16384 | unchanged (retokenizing 5B+ tokens is a week of work) |
| `d` | 2048 | 2560 | width bump ×1.25 |
| `n_neurons` | 4864 | 6144 | width bump ×1.26 (keeps ratio ~2.4) |
| `d_mem` | 2048 | 2560 | width bump matches `d` |
| `n_layers` | 5 | 8 | depth ×1.6 (larger lever for capability) |
| `block_size` | 640 | 640 | keep (VRAM probe may allow more; don't overreach) |
| `batch_size` | 14 | **8 (probe first)** | see Cost & Risk |
| `grad_accum` | 2 | **4 (probe first)** | eff. batch 32 |

**Rationale for going deeper (5L→8L) vs. wider-only:**
- 500M v3's memory-pathway signal is stronger with depth than with pure
  width (mem-Δ +298 at 5L; the deep 90M runs measured relative benefit of
  depth). GDN-2 chunk of `d_mem`-shape is repeated per layer, so 8 layers
  gives 8 separate memory buffers with content composition between them.
- 8 layers is the depth at which Engram-v2's per-block bucket lookup starts
  paying off (heads=4 was chosen at 4L/5L; at 8L may want a re-screen of
  heads=4 vs heads=6 at 90M, but not blocking).
- Existing chunk-kernel gradient-checkpointing (`chunk=32`) should scale
  linearly with depth without new work.

**Parameter-count dry-run**: `python -c "from src.train_gpu import
build_model; import yaml; c = yaml.safe_load(open('configs/1b_v1.yaml'));
m = build_model(c); print(sum(p.numel() for p in m.parameters())/1e6, 'M')"`.
Must land 900M–1100M or shape gets tweaked before launch.

**Recipe fields to CARRY FORWARD from `500m_v3.yaml` unchanged** (in order
of appearance in the source file):

```
arch: spiking            recurrent: false        recurrent_depth: 1
use_fpt: true            fpt_K: 10               use_fla_kernel: false
fused_ce: true           delta_rule: true        decay_gate: true
beta_floor: 0.1          mqar_aug: true          mqar_N_min: 16
mqar_N_max: 16           mqar_weight: 0.5        sparsity_target: 0.05
gdn2_enabled: true       gdn2_chunked: true      osdn_enabled: false
engram_v2_enabled: true  engram_v2_K: 8192       engram_v2_heads: 4
engram_v2_orders: [2,3]  engram_v2_rank: 64      curriculum_enabled: true
curriculum_frac: 0.1     curriculum_block_tokens: 4096
optimizer: muon          normuon: true           normuon_beta2: 0.95
muon_lr: 0.005           normuon_eps: 1.0e-8
muon_fix_gate_matrices: true    muon_route_embeddings_by_type: true
use_rmsnorm: true        embed_scale: true       z_loss_weight: 0.0001
mtp_enabled: true        mtp_offset: 2           mtp_weight: 0.3
mtp_depth: 2             lr: 0.0004              lam: 0.98
snapshot_brain: false    amp: true               compile: true
compile_mode: default    compile_dynamic: false  seed: 0
```

**Recipe fields to SET FROM PROBE / recalibrate for 1B:**

- `sparsity_lambda`: start at 1.0 (500M v3 endpoint value after mid-run bump);
  reconsider only if a 90M-at-1B-shape screen shows spike-creep signature.
  The `sparsity_cleanup_*.yaml` ratio (6× over pretrain endpoint) carries
  forward AS A RATIO — [[neuro-sparsity-refine-ratio-gotcha]] burned us
  on this exact confusion at 500M.
- `keep_last: 4` (was 8 at 500M v3 — checkpoint size ~doubles at 1B; 4 is
  enough for PMA + rollback safety).
- `max_tokens: 10_000_000_000` (10B tokens, 10 tok/param). Bump to 15B only
  if wall-clock allows and the eval gate (§9) has room to improve.
- `eval_every`: keep 1000 (~91M tokens/eval at 91k tok/step; ~110 evals
  over 10B tokens — plenty).
- `ckpt_every`: 1000 same (checkpoint at every eval; PMA-friendly).

---

## 6. Data + tokenizer

**Tokenizer**: unchanged. `vocab: 16384` Llama-3-derived tokenizer, same as
500M v3.

**Pretrain data**: unchanged. `/home/glenn/projects/bdh/data/tokenized/
train.bin` — enough for at least 15B tokens (the 500M v3 used 5B of it; the
BDH-era tokenization was sized for a much larger training budget). Confirm
file size covers `1.2 × max_tokens × 2 bytes/token` before launch.

**Curriculum warmup**: same source, `curriculum_order_4096.npy`
(configs/500m_v3.yaml:95, data/curriculum_order_4096.npy). Reused verbatim.

**SFT data**: `yahma/alpaca-cleaned`, unchanged from
`sft_500m_full_v3.yaml:25`. At 1B, `sft_max_len: 512` still appropriate for
Alpaca. `sft_500m_full_v3.yaml` line 15's "~8 Alpaca epochs at 512-tok cap"
is roughly matched by `max_tokens: 500_000_000` — carry forward.

**DPO data (if used)**: `argilla/ultrafeedback-binarized-preferences-cleaned`
(the same 60k-pair set 500M v3 overfit on). If DPO is used at 1B, hard-cap at
0.5 epochs (~30k pairs, ~60M tokens) with `gen_check()` early-stop, or use
a 5× larger preference set. Recommendation: **do not use DPO at 1B v1** —
skip straight from SFT to sparsity cleanup and demo. See [§8](#8-post-pretrain-pipeline).

---

## 7. Hardware plan

**Primary: 5090 (32 GiB)** — pretrain runs here for ~25–30 days uninterrupted.
- Batch/grad_accum from live probe (§4).
- Compile mode: `default` (not `reduce-overhead`; the CUDA Graph bug at
  500M v3 was upstream PyTorch and stays open at 1B, see 500m_v3.yaml:136).
- WSL config: keep 24 GiB WSL RAM cap ([[reference-wsl-memory-config]]).
- Disable Windows update auto-reboot for the pretrain window
  ([[reference-wsl-auto-upgrade-kill]]).

**Secondary: 5080 (16 GiB)** — idle during pretrain (nothing to do without a
base ckpt). Can run 30M/90M pre-launch screens for the sparsity-λ retune or
Engram-v2 sizing check. After pretrain: runs post-pretrain screens in
parallel with 5090's SFT.

**Storage**:
- Each 1B ckpt at fp32 (2 optimizers state + weights + MTP + Engram-v2
  buckets) ≈ 14–18 GiB.
- With `keep_last: 4` = ~72 GiB peak in `models/checkpoints/1b_v1/`.
- Full pipeline (`1b_v1` + `sparsity_refine_1b_v1` + `sft_1b_v1` +
  `sparsity_cleanup_1b_v1`) with `keep_last: 3` each = **~200 GiB during the
  active month**. Check disk headroom before launch.

---

## 8. Post-pretrain pipeline

Direct port of the 500M v3 pipeline with the DPO stage removed by default.

```
1b_v1 (10B tok pretrain)
        │
        ▼
sparsity_refine_1b_v1  (150M tok, 3× ratio over pretrain endpoint λ)
        │
        ▼
sft_1b_v1              (500M tok, Alpaca full-FT, LR = pretrain LR / 5)
        │
        ▼
sparsity_cleanup_1b_v1 (265M tok, 6× ratio over pretrain endpoint λ)
        │
        ▼
Pass 1 endpoint = demo artifact
```

**Why DPO is dropped from the default path** ([[neuro-500m-v3-final-verdict]]
and [[neuro-simpo-collapse-2026-07-26]]): both preference-optimization attempts
at 500M failed (SimPO×2, standard DPO), and the failure mechanism (overfitting
under repeated passes over a small preference set) gets *worse* at higher
capacity, not better. The 500M v3 demo artifact is post-SFT + post-sparsity-
cleanup only — that's the pattern to inherit unless we build a DPO safety net
first.

**If DPO is desired for the 1B demo:**
1. Pre-launch: run a `dpo_30m` screen with `gen_check()` early-stop enabled
   and a 5× larger preference set (or `dpo_beta` = 0.3, higher regularization).
   Only if this screen shows real behavioral improvement over SFT-only,
   promote to a 1B DPO stage with `max_tokens ≤ 60M` (0.5 epoch).
2. Otherwise, ship SFT-only per the 500M v3 pattern.

**Sparsity λ ratio (critical lesson from 500M v3)**:
[[neuro-sparsity-refine-ratio-gotcha]] — the 500M v3 first sparsity refine
undercompressed because it used the *absolute* λ that worked at 30M rather
than the *ratio* over the pretrain endpoint's λ. At 1B, if the pretrain
mid-run tunes `sparsity_lambda` upward (as 500M v3 did, 0.5→1.0), the refine
config MUST use `pretrain_endpoint_λ × ratio`, not a hardcoded number
copied from a screen.

---

## 9. Evaluation protocol (pre-registered)

Every metric below is measured on the **final Pass-1 sparsity-cleanup
endpoint** unless noted. Baseline is 500M v3's Pass-1 endpoint
(sparsity_cleanup_500m_v3/step_0016388.pt).

### 9a. Quantitative — internal (existing scripts)

- `scripts/ppl_eval.py` — val_ppl on held-out FineWeb-Edu slice.
  **Target: < 40** (500M v3 = 43.3). Miss target = investigate; do not ship
  without an understanding of why.
- `scripts/recall_eval.py --replay split` — MQAR recall at N=4/16/64.
  **Target: ≥ 90/50/25%** (500M v3 = 89/47/22%). Recall at N=64 is the
  hardest number and the honest scale test for the Hebbian pathway.
- `scripts/engram_factual_eval.py` — Engram log-prob delta (ON vs OFF).
  **Target: consistent positive delta at every eval** (proves the memory
  scaffold still helps at 1B, not just at 30M/500M).
- Spike rate (in-training telemetry). **Target: ≤ 0.15 at Pass-1 endpoint**
  (500M v3 hit 0.110). Fails-loud if spike creeps > 0.20 without a λ bump.
- β `dead_frac` = 0% on all blocks at Pass-1 endpoint (500M v3 = 0%). Hard
  gate — a dead block is a broken memory path, not a stylistic issue.

### 9b. Quantitative — external (community-standard benchmarks)

Run via `scripts/lm_eval.py` (EleutherAI harness wrapper, existing).
**Baseline to beat**: 500M v3 Pass-1 endpoint scored on the same subset.
This is the *demo-relevant* baseline; the honest comparison is to community
1B-class open models, but running those in-house is out of scope for this
pretrain — reference them from published numbers in the demo write-up
instead.

Subset (kept short for iteration cost):
- **HellaSwag** (0-shot) — commonsense; standard 1B-open-model report.
- **ARC-Easy** (0-shot) — factual reasoning; 1B-scale-friendly.
- **ARC-Challenge** (0-shot) — factual reasoning; harder.
- **PIQA** (0-shot) — physical commonsense.
- **WinoGrande** (0-shot) — coreference.

MMLU is *excluded* from the 1B v1 default eval set: our tokenizer/data lack
the QA-instruction density MMLU rewards, and reporting a poor MMLU score
on a demo model buries the recall/memory story. If a community bench needs
MMLU, run it once as a reference and note the limitation.

**Success bar (pre-registered)**: ≥ 3 of 5 subset benchmarks show a real
improvement over the 500M v3 Pass-1 endpoint (absolute Δ ≥ 2pp or clear
significance, not within-noise). Falling below this bar means "the scale-up
didn't translate to capability" and the demo pitch has to change.

### 9c. Qualitative — chat/demo readiness

Run `~/projects/neuro/chat.sh 1b_v1_sparsity_cleanup_pass1` on the
soon-to-be-curated REPL prompt list (per the handoff's next-session task #1).
Judge:
- **Golden-path prompts** (well-formed instructions, factual recall):
  coherent, direct, no repetition-lock.
- **Follow-up prompts** (multi-turn): remembers previous turn context,
  extends without contradiction.
- **Edge cases** (incomplete-sentence prompts, ambiguous instructions):
  degrades gracefully; no obvious tokenizer glitches.

**Success bar**: qualitative pass at least as strong as 500M v3 Pass-1
(baseline the two side-by-side on the same prompts using the two
chat.sh runs).

### 9d. Persistent-brain demonstration (the unique lever)

The project's unique differentiator is the persistent Hebbian brain state.
For a demo, we need at least one prompt sequence where:
- Prompt A "teaches" the model something (e.g., a novel definition, a
  fictitious fact).
- Prompt B, 5+ turns later, tests whether the model can retrieve A without
  it being in the recent context window.
- 1B model should demonstrate this cleanly; 500M v3 does at short horizons.

This is *not* a quantitative eval — it's a demo capability. But it needs to
be verified working before the 1B ships, because it's the pitch.

### 9e. Known failure modes to design the eval around

- **Batch-A build() drift** ([[neuro-train-sft-build-drift-fix]] and prior
  three recurrences) — every eval script MUST delegate to
  `train_gpu.build_model`, not maintain a local `build()`. Confirmed for
  `chat.py`, `recall_eval.py`, `engram_factual_eval.py`,
  `train_sft.py`, `train_dpo.py` after the 2026-07-20 fix pass. Any *new*
  eval script written for 1B must follow the same pattern.
- **Eval-noise floor**: from [[reference-eval-noise-fix]], the fixed-seed
  100-iter eval config brings ppl noise below ~1.2 ppl. Any ppl delta
  claimed must exceed 2 ppl to count. Legacy pre-fix numbers are not
  comparable.
- **Recall vs. general-capability tradeoff**: 500M v3 saw this at
  mqar_cadence=2 (general ppl improved, recall crashed at high N). At 1B,
  if a capability lift comes at the cost of recall regression at N=64,
  that's the same failure mode — do NOT ship.
- **Preference-optimization overfitting** — if DPO is used, `gen_check()`
  every 500 steps and stop at first sign of repetition drift, per journey
  §46 pattern.

---

## 10. Recommendation & decision gate

**Recommendation: scope now (this doc), do NOT launch yet.**

Three pre-launch gates must clear before the 1B pretrain starts. All three
can be worked in parallel — none needs a checkpoint from another to begin.

### Gate 1: 500M v3 demo verdict (blocks 1B launch)

Curate the REPL prompt list per handoff task (1) and run through
`chat.sh sparsity_cleanup_500m_v3`. Answer the question: **is the 500M v3
already demo-worthy?**
- If **YES** — 1B is a "same recipe, bigger, better" pitch. Launch after
  gates 2 and 3 clear.
- If **NO, and the failure is capability** (facts wrong, reasoning broken)
  — 1B likely helps. Launch after gates 2 and 3.
- If **NO, and the failure is chat pipeline / SFT template / brain-mod
  immaturity** — 1B will inherit the same problems. **DO NOT launch until
  those are fixed at 500M scale first.**

Owner: this is the first task of the next session; not gated on any 1B work.

### Gate 2: 90M same-recipe reconfirm at 1B-shape (blocks 1B launch)

Build `configs/screen_1b_v1_90m.yaml`: same recipe as 1b_v1 (d=2560,
n_neurons=6144, d_mem=2560, n_layers=8) but scaled to 90M-param equivalent
(reduce depth or width to hit 90M ± 10%; keep the aspect ratios). Run one
seed at 90M tokens (~2.5h on 5090). Compare against 500M v3's own 90M
de-risk (bake_normuon_engramv2_combined_90m) — the recipe should behave
equivalently at the new shape ratios.

Success bar: val_ppl and mem-Δ within 10% of the 500M v3 90M numbers,
β dead_frac 0%, spike rate on trajectory to plateau.

Owner: 5080 or 5090 idle slot. Runtime: ~2.5h. Blocks: 1B launch.

### Gate 3: parameter-count + VRAM probe on true 1B shape (blocks 1B launch)

- `python -c "from src.train_gpu import build_model; import yaml;
  c = yaml.safe_load(open('configs/1b_v1.yaml')); m = build_model(c);
  print(sum(p.numel() for p in m.parameters())/1e6, 'M')"` — must return
  900–1100 (target 1000). If off, tweak shape and re-run.
- Live VRAM probe at true 1B shape (matches 500m_v3.yaml:71 methodology):
  smoke run at batch=8/grad_accum=4/block_size=640, watch nvidia-smi for
  dedicated + shared VRAM. Pick the largest `batch_size` that fits
  entirely in dedicated (no WDDM spill). Set `batch_size:` in
  `configs/1b_v1.yaml` from probe, not from this doc's estimates.

Owner: 5090 idle slot. Runtime: ~1h (probe + smoke train). Blocks: 1B launch.

### After all three gates clear

Launch `configs/1b_v1.yaml`. Expected wall-clock: 22–32 days for the full
pipeline through Pass-1 sparsity cleanup. Set explicit calendar reminder to
re-check the run every 3–4 days (Windows reboot, WSL upgrade, VRAM spill —
[[reference-wsl-auto-upgrade-kill]] and journey.html §45 all three scares).

### Not doing (deferred additions)

Explicitly not in scope for 1B v1:
- **Per-token MoR router** — blocked on sparse-gather kernel that was
  never built (see 500m_v3.yaml:33–40 and to-do.html Phase 3).
- **SDM** — CLOSED-NEGATIVE at 30M composition ([[neuro-sdm-closed]]).
- **Fracconn** — CLOSED-NEGATIVE ([[neuro-fracconn-built]]).
- **DeepSeek Sparse Attention** — we don't have attention layers to
  attach it to; would be a Phase 4 architecture rewrite.
- **KD from a larger teacher** — was in demo_phase1b_500m_v2_fast but
  dropped for v3 recipe; folding it back in would be a separately-screened
  enhancement, not part of the scale test.
- **Confidence head / Markov / MTP-spec-decode inference** — inference-time
  add-ons, orthogonal to pretrain, land AFTER 1B ships as separate features.

---

## References

Ranked: primary port targets first, then eval-protocol / rigor sources,
then related / lineage work.

**Primary port targets** (this recipe is already in-house; the 1B is a
scale test of settled architecture — no new arch to port):
- `configs/500m_v3.yaml` — the exact recipe being scaled. Every field
  decision has a paper-trail in that file's header comments and in
  `docs/journey.html` §44–47.
- `configs/sft_500m_full_v3.yaml` — SFT stage template.
- `configs/sparsity_cleanup_500m_v3.yaml` — sparsity cleanup template
  (**note the ratio-not-absolute lesson**, [[neuro-sparsity-refine-ratio-gotcha]]).
- `docs/gdn2_osdn_design.md` — GDN-2 core memory-rule; unchanged at 1B.

**Eval protocol / rigor sources**:
- EleutherAI lm-evaluation-harness — https://github.com/EleutherAI/
  lm-evaluation-harness. `scripts/lm_eval.py` already wraps it. Community-
  standard baseline for the external quantitative eval subset.
- Chinchilla scaling law (Hoffmann et al., arXiv:2203.15556) — 20 tok/param
  as compute-optimal. 1B v1 targets 10–15 tok/param, undertrained per
  Chinchilla but consistent with 500M v3's own ratio and honest about the
  compute constraint.
- "Reusing Overtrained Language Models Saturates Scaling"
  (arXiv:2510.06548) — evidence that heavy overtraining past Chinchilla
  gives diminishing returns; supports the 10–15 tok/param choice for a
  demo model rather than 60,000:1 like Qwen3-0.6B.
- 500M v3's own eval-noise-floor discipline
  ([[reference-eval-noise-fix]]) — the 2 ppl "real delta" threshold used
  in §9a comes from this.

**Related / lineage work** (2026 SOTA that validates the recipe direction,
not things we'd port from):
- MiniMax M2 series technical report — DeepSeek Sparse Attention + Gated
  DeltaNet in a hybrid stack (2026). Validates GDN-2 as the right family.
  Not portable directly (they use attention layers we don't have).
- Qwen 3.5 Next — GDN layers in a dense + MoE hybrid. Same family
  validation.
- Instella (arXiv:2511.10628) — "Fully Open Language Models with Stellar
  Performance" — recipe transparency comparison point for the demo
  write-up (not to port from).
- INTELLECT-1 (arXiv:2412.01152) — decentralized 10B open training
  reference for a community-open-model comparison.

**Failure modes / anti-patterns cited above**:
- [[neuro-500m-v3-final-verdict]] — DPO discard rationale.
- [[neuro-simpo-collapse-2026-07-26]] — preference-optimization failure
  patterns to design DPO safety around.
- [[neuro-sparsity-refine-ratio-gotcha]] — ratio-not-absolute lesson.
- [[neuro-train-sft-build-drift-fix]] — every new eval script must
  delegate to `train_gpu.build_model`.
- [[reference-wsl-auto-upgrade-kill]] and
  [[reference-wsl-memory-config]] — unattended-run hazards.
- [[neuro-lora-at-30m-plateau]] — SFT stage uses full-FT, not LoRA.
- [[neuro-beta-collapse-fix]] — `beta_floor: 0.1` is required.
- [[neuro-normuon-renorm-fix]] — NorMuon global-renorm bug (already
  fixed in-tree, mentioned so a future reader knows why NorMuon is safe).
