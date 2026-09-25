# Pre-flagship polish recommendations — 2026-08-07

Review of the last few days' work + recipe/code audit + optimization
recommendations across model-quality axes. For Glenn's morning read;
recommendations are ranked by expected value vs cost.

Companion to `docs/plan_2026_08_05.md`. Some recommendations here would land
on top of the composed recipe (arm B + W2's λ=0.5 + W4's grad_accum=4)
currently running as W5→P4-seed-0→P4-seed-1 on the 5090.

---

## Wake-up TL;DR

- **The composed recipe won decisively.** W5's recall at step 4000 (~131M tok) = **100/100/97/91/41%** at N=4/8/16/32/64 — sharpest recall this project has produced. Even at partial training, crushes arm B (100/98/89/66/27) and W4 (100/100/95/80/33).
- **P4 seed 0 is running (5090)**, compiling as of 01:15:35. Chain will auto-launch seed 1 when seed 0 completes (~15:15 today). Total chain complete ~05:15 tomorrow.
- **W5's SIGINT-graceful-save didn't fire visibly** (no handler-output line, no fresh ckpt); root cause partly diagnosed (see incident section below), P4 seeds carry the same risk but periodic ckpts every ~2h limit worst-case loss to 2h of tail — acceptable.
- **Tonight's code changes** (NaN guard, vocab encoder/decoder consistency, brain_domain_transfer plumbing) are all landed and unit-tested. P4 seeds pick them up automatically.
- **New polish backlog** for the flagship: `-k 300` grace, skip end-of-run eval under SIGINT, MQAR range parametrization, `grad_clip: 1.0` explicit in configs, plus the earlier list (W_beta/W_alpha skip, spike aggregation, quadratic λ, dead-code strip).

---

## Overnight news (2026-08-07 ~01:15 UTC-7)

- **W5 recall on step_4000 ckpt (partial training, 131M tokens): 100/100/97/91/41% at N=4/8/16/32/64.** BEST recall this project has produced. Crushes arm B baseline (100/98/89/66/27) and W4 (100/100/95/80/33) at LESS training. Composition of W2+W4 is a decisive win.
- Chain auto-passed the recall gate at 01:15:35 and launched **P4 seed 0** immediately. Seed 1 chains after.

### Incident: W5 SIGINT-graceful-save appears to have failed

W5 exit code 130 (matches SIGINT), but log shows no `[SIGINT] received` handler output and no fresh ckpt on disk (only the periodic step_4000 from Aug 6 21:42). Two possible root causes, still ambiguous:

1. **`-k 60` too short**: post-loop sequence is `_save_ckpt` (~15s) + `evaluate` at end (100 iters, ~50s at W5 throughput) + `energy.compare` (~1s) ≈ 66s, just over the 60s SIGKILL grace after SIGINT. SIGKILL would then be exit 137, not 130 — so this hypothesis has a hole.
2. **Kernel-buffered log writes lost**: WSL filesystem cache may not have flushed handler output before SIGKILL. Would explain missing log lines even if handler ran.

**Immediate mitigation for P4 seeds** (already committed to `-k 60`): periodic `ckpt_every: 4000` + `keep_last: 2` means worst-case loss is ~2h of tail training per seed. Acceptable for a 14h run.

**For the flagship**: bump `-k` grace to 300s (5 min) — comfortably covers save+eval+energy. Also consider skipping the end-of-run `evaluate + energy.compare` under SIGINT (only do them on natural completion). Both cheap changes.

---

## APPLIED TONIGHT (2026-08-07 review pass)

Landed as verified code changes with unit tests; P4 seeds will pick these up
automatically when they launch (fresh Python process each). W5 is unaffected
(loaded module at start).

| item | files | test |
|---|---|---|
| **NaN/Inf guard on grad norm** — skips optimizer.step() on non-finite grads, aborts with exit 3 after 3 consecutive (prevents silent weight corruption; existing dstd watchdogs cannot detect NaN because `nan > threshold` is False) | `src/train_gpu.py` | `temp/test_nan_guard.py` — 6/6 pass |
| **Vocab encoder/decoder consistency** — encoder preserves id 0 (matches training), decoder emits `<unk>` for id 0/OOV instead of silent drop | `scripts/generate.py`, `scripts/lm_eval.py`, `scripts/chat.py` | `test_builders.py` — 6/6 pass; lm_eval delta within noise; generate now shows `<unk>` where content was silently dropped |
| **brain_domain_transfer `--run` arg** — was hardcoded to `phase3_5090`, now works with any checkpoint dir; unlocks plasticity axis measurement | `scripts/brain_domain_transfer.py` | syntax verified, delegates to `_canonical_build` |

## Bug backlog found in this review — NOT YET APPLIED

| item | severity | fix cost |
|---|---|---|
| `kd_teacher_topk` set in 2 KD configs but never read by code (dead knob) | low | rm from configs (KD is closed-negative anyway) |
| `MQAR_{KEY,VAL}_{LO,HI}` hardcoded to 16k-vocab range in `train_gpu.py`; also hardcoded in `recall_eval.py`. Recall metric only exercises student ids [1000..14000] — leaves ids [14000..32767] untested at 32k vocab | medium (methodology) | parametrize on `c["vocab"]`, add config knobs |
| `grad_clip` never set explicitly in any config (uses default 1.0) — reproducibility drift risk | low | add `grad_clip: 1.0` to all recipes |
| Autocast disable inside GDN-2 chunked kernel + Inductor `emulate_precision_casts` — verified handled correctly, but any future kernel change here needs a NaN smoke test | none (documented risk) | note in the kernel comment |

---

## TL;DR — top 5 pre-flagship actions

1. **Fix vocab encoder/decoder train/test mismatch** (30 min code) — 4% of
   training tokens map to student id 0 (unknown), but `generate.py` and
   `lm_eval.py` DROP those tokens at inference instead of preserving them.
   The model was trained on sequences that include id 0; inference gives it
   a different distribution. Real fix, easy.
2. **Skip `W_beta`/`W_alpha` construction under gdn2** (10 min) — dead
   params waste embed_grad state, confuse diagnostics. Pre_350m gotcha #7.
3. **Fix spike-rate aggregation dilution (not just log)** (30 min +
   sparsity_target retune to 0.171) — the training loss currently uses the
   diluted mean, so the sparsity penalty is applied to the wrong number.
   Prevents scale-transfer confounds when attention ratio changes.
4. **Quadratic sparsity penalty** (5 lines + retune λ) — self-corrects
   penalty strength with model scale. Pre_350m gotcha #4. Optional but
   flagship-friendly.
5. **Add MTP correctness spot-check** — MTP loss can silently disable under
   torch.compile (documented in model.py comment lines 1291-1298). No known
   issue on current recipe, but worth verifying on arm B's checkpoint.

The composed recipe (P4 seed 0/1 running now) doesn't need any of these to
be a valid 2-seed variance check. These land on top before the 363M
flagship.

---

## 1. What we know across quality axes (arm B baseline, 131M tokens)

Measured this week:

| axis | metric | arm B result | recipe assessment |
|---|---|---|---|
| **memory & recall** | recall_eval `--replay split` N=4/8/16/32/64 | 100/98/89/66/27% | ✅ project differentiator, strong |
| **coherence** | generate.py 5 prompts | fluent, topic drift, "of theity" artifact | ⚠️ 4% vocab drop artifact + heavy undertraining |
| **generative quality** | subjective | early-pretrain, no SFT/DPO yet | ⚠️ SFT is post-pretrain, not blocking |
| **stability** | 5.4h continuous run | zero divergence, dstd 0.9-1.4 flat | ✅ Muon-wd + gate-norm holding |
| **plasticity** | brain_domain_transfer | not measurable (hardcoded to phase3_5090) | — needs plumbing fix |
| **efficiency / sparsity** | energy report | 1.99× @ seq 512, 2.74× @ seq 4096 (matched depth) | ✅ improves with scale |
| **robustness** | lm_eval N=300 | HellaSwag 30%, ARC-E 32%, ARC-C 17% | ⚠️ near-chance, ~4% biased by vocab drop |

Composed recipe (W5 → P4) is projected to improve on all axes at similar
token budget: W4's grad_accum=4 alone dropped val_ppl to 38.95 at 216M tokens.

---

## 2. Bugs found in this review

### 2.1 Vocab encoder/decoder train/test mismatch  **[recommend fix pre-flagship]**

**Problem.** `vocab_map_32768.npy` is a frequency-based Llama-3 → student-32k
mapping. 74.5% of Llama-3 tokens (95,489 / 128,256) are rare and map to
student id 0. In real training data this affects **~4.0% of tokens** (~1 in
25) — including common English content tokens like `capital` and `machine`
that happen to fall outside the top-32k by count.

**Bug.** Different behavior between training and inference:
- **Training** (`data_stream.py:37`): `block = self.lut[raw_block]` —
  keeps id 0 in the sequence (model learns "id 0 is a legitimate rare
  token").
- **Inference** (`generate.py:51`, `lm_eval.py:52`):
  `[int(lut[t]) for t in ids if lut[t] != 0]` — **DROPS** id 0 from the
  sequence entirely.

So an inference prompt containing "the capital of France" arrives at the
model as "the of France" (missing "capital"). Model sees a subtly
different distribution than it was trained on. Sequences are ~4% shorter
at inference than at training on average.

**Impact.** Directly explains the "of theity" and missing-content-word
patterns in arm B's generate output. Also biases lm_eval accuracy
(HellaSwag N=300 = 30% vs 25% chance; length-normalized scoring is affected
by different drop rates per continuation). Val_ppl is unaffected (both
sides use the same `data_stream.py` LUT application).

**Fix** (safe, no retraining needed):
```python
# generate.py, lm_eval.py load_tokenizer() — remove the drop filter
def encode(text):
    ids = tk.encode(text, add_special_tokens=False)
    return [int(lut[t]) for t in ids]   # keep id 0, match training

def decode(ids):
    parts = []
    for i in ids:
        if 0 < i < vocab and inv[i] >= 0:
            parts.append(tk.decode([int(inv[i])]))
        else:
            parts.append("<unk>")   # visible marker, no silent drop
    return "".join(parts)
```

**Verify before flagship** with a quick `lm_eval` re-run on arm B — expect a
small but real accuracy lift on tasks with rare content words (ARC-C
especially benefits, since it has more technical vocabulary).

**Longer-term fix** (out of scope for flagship): larger vocab (e.g., 64k)
or fresh byte-fallback BPE — see `data/vocab_map_32768_freshbpe.npy`
placeholder. Would require rebuilding LUT and re-evaluating training
efficiency; not for this generation.

### 2.2 `sft_500m_tulu3` uses stale kwargs (deferred)

Referenced by `chat.py` but the checkpoint precedes several arch flags.
Not blocking flagship. `test_builders.py` didn't cover it because it's not
in the drift-bug script list. Consider adding to P2 test-builders
guardrail.

---

## 3. Recipe recommendations for the 363M flagship

Starting from the composed recipe (W2+W4 winning combo). All numbers
assume the P4 2-seed variance check passes.

### 3.1 KEEP (no change)

These are settled and shouldn't be touched:
- **`gdn2_gate_norm: true`** — required (W1 proved: watchdog aborted in
  ~30 min without it)
- **`weight_decay: 0.1`** — root-cause stability fix (Muon+AdamW split)
- **`sparsity_lambda: 0.5`** — W2 rank-quality win over 1.0
- **`grad_accum: 4`** (effective batch 64) — W4 accuracy win
- **`beta_floor: 0.1`** — settled since 2026-07-10 (β collapse fix)
- **`attention_layer_positions: [3]`** (7:1) — decisively beat 3:1 on
  persistent recall (phase-1 finding)
- **`attention_qk_norm: true`** — 3-seed variance fix
- **`engram_v2_enabled: true`** with `K=8192, heads=4, orders=[2,3], rank=64`
- **`mtp_enabled: true, mtp_offset=2, mtp_weight=0.3, mtp_depth=2`**
- **`compile: true, compile_mode: default`** — validated (never
  reduce-overhead — open PyTorch bug)
- **`use_fpt: true, fpt_K: 10`** — parallel forward, ~3× speedup
- **`gdn2_chunked: true`** — 23× speedup over sequential loop

### 3.2 SHAPE — recommend

- **`d=1280, n_neurons=3072, d_mem=1280, n_layers=9`** — 363M params, clean
  ~350M target from plan doc §Phase 5. Head_dim 128 (attention_heads=10).
  Aspect ratio n≈2.4d matches 500M v3 and the 1B design.
- **Attention position `[4]`** (position 4 in 9-layer stack) — maintains 7:1
  ratio, centered.

### 3.3 CONTEXT LENGTH — probe first

- Currently `block_size: 512`. Plan doc P5 recommends probing 768/1024 for
  the flagship.
- **Recommendation:** VRAM-probe **before** the flagship-shakedown launch. Both
  memory-recall and matched-depth energy advantage grow with context length
  (2.74× at seq 4096 vs 1.99× at seq 512). Cheap win if it fits.
- Watchdog: gate-norm's activation memory + block 768 may push toward
  WDDM spill on the 5090's 32 GB. See `neuro-500m-v3-vram-sweetspot`.

### 3.4 LR — data-informed

- Composed recipe uses `lr: 5e-4` at 276M/1.1B tokens (~0.5× Step-Law
  optimum). W3's `lr: 7e-4` REJECTED at 276M — worse rank quality despite
  higher memory-Δ.
- At 363M/7B tokens (Chinchilla-ratio, plan target), Step Law suggests
  `η(N,D) = 1.79·N^(-0.713)·D^(0.307)` ≈ **6.6e-4**. Direction is up
  slightly.
- **Recommendation:** `lr: 6e-4` for flagship. Middle ground —
  0.9× Step-Law, safer than W3 was. Watch dstd through the 5h shakedown
  before committing to full 7B tokens.

### 3.5 SPARSITY — retune target for dilution fix

If we land the aggregation fix (recommendation §4.2), the training-time
sparsity penalty starts acting on the TRUE spiking-layer rate instead of
the diluted mean. To preserve behavior at the crossover point:
- Old: `sparsity_target: 0.15` on diluted mean → true target ~0.171
- New: `sparsity_target: 0.171` on true mean → same effective penalty
- Or: keep target 0.15 on true mean → penalty is ~14% stronger (spike will
  land lower). May be fine given W2 evidence that spike went 0.15 → 0.19
  at λ=0.5 without harm.

Recommendation: **retune target to 0.171** to preserve composed-recipe
behavior; make the aggregation fix isolate cleanly from the sparsity
question.

### 3.6 ENGRAM v2 — consider K bump

- Current: `engram_v2_K: 8192, heads: 4, orders: [2, 3], rank: 64`
- Larger model → more capacity → could support larger bucket count without
  interference
- 500M v3 used `K: 16384` — worked; scaled recall linearly
- **Cheap experiment:** could 30M-screen K=8192 vs K=16384 on the 5080
  tonight (~3h, one variable), but it's low-priority given recall is
  already strong. Not blocking.

### 3.7 CURRICULUM — probably skip

- `curriculum_enabled: false` currently
- The 500M v3 recipe used it (curriculum warmup)
- Not on flagship critical path — separate 30M A/B, low priority

### 3.8 CHECKPOINT DENSITY — bump for flagship

- Currently `ckpt_every: 4000` in P4 seeds (~2h wall clock between ckpts)
- Flagship at 7B tokens will be ~7 days. `keep_last: 6` and
  `ckpt_every: 8000` is a reasonable trade — save every ~4h, retain last
  24h of forensics.

---

## 4. Code hygiene recommendations (pre-flagship)

### 4.1 Skip `W_beta`/`W_alpha` construction under gdn2  **[10 min, low risk]**

Pre_350m gotcha #7. Under `gdn2_enabled + decay_gate`, `W_beta` and
`W_alpha` are constructed but receive zero gradient (superseded by
`W_b`/`W_w`/`W_fdecay`). They:
- Consume optimizer state (Muon momentum, NorMuon second moment)
- Are visible in state_dicts (waste storage, checkpoint size)
- Confused a diagnostic session earlier this week (30 min lost)

Guard in `model.py` block constructor: `if not gdn2_enabled: self.W_beta =
...`. Verify state_dict compat with a strict-key load test before
promoting (`test_builders.py` will catch it).

### 4.2 Fix spike-rate aggregation dilution  **[30 min, minor sparsity retune]**

Pre_350m gotcha #5. Currently `src/model.py:1312`:
```python
spike_rate = torch.stack([s.mean() for s in all_spikes]).mean()
```
`all_spikes` contains attention-layer zero-stub tensors. At L=8 with 1
attention layer, reported rate = 0.875 × true rate. My earlier fix landed
a diagnostic log line; the FIX is to change the aggregation itself.

Change to average over spiking-block entries only. Then the sparsity
penalty acts on the true rate, and `sparsity_target` needs adjustment (§3.5).

### 4.3 Quadratic sparsity penalty  **[15 min, opt-in]**

Pre_350m gotcha #4. Currently `train_gpu.py:541`:
```python
main_loss + λ * (sr - target).clamp(min=0)          # linear hinge
```
Gradient w.r.t. spike is constant λ regardless of overshoot magnitude.
Quadratic hinge `λ * (sr - target).clamp(min=0)**2` gives
gradient `2λ(sr - target)` — self-corrects, better scale-transfer.

**Not urgent** — W5 evidence shows the linear penalty holds fine at 276M.
But adding a flag `sparsity_quadratic: false` and testing on flagship as an
optional lever is cheap insurance. Recommend adding the flag but leaving
false for the flagship's first pass.

### 4.4 Dead-code strip (P6 from plan)  **[1h, before flagship's 7-day commit]**

`docs/plan_2026_08_05.md` Phase 4 mentions:
> strip dead SDM/OSDN/Titans/Markov/MoR/confidence/Engram-v1 branches
> from model.py

Every dead branch is:
- A torch.compile cache footgun (may pull in dead paths during graph
  capture, mis-cache, or trigger fallback)
- Diagnostic noise (developer time lost investigating a code path that's
  actually dead)

Byte-verify with `temp/test_builders.py` on a canonical checkpoint before
promoting. Should land **before** the flagship's 7-day commit; a compile
regression 3 days into a 7-day run is expensive to catch.

### 4.5 P2 instrument repair extension — brain_domain_transfer  **[30 min]**

`brain_domain_transfer.py` was delegated (build_from_ckpt uses
`_canonical_build` now) but `CK_DIR` is still hardcoded to `phase3_5090`.
Add `--run` argument, then re-run on arm B for the missing plasticity
axis baseline. Also add to `test_builders.py`.

---

## 5. Not doing — deliberate

- **KD from teacher.** Rejected in the redesign investigation (3 alphas
  all worse). No revisit.
- **SDM revival.** CLOSED-NEGATIVE at 30M composition. Do not re-launch.
- **Fresh 128k vocab.** Would require re-tokenizing 58 GB of train.bin
  and adding ~120M params to embedding table (at d=1280, that's 33% of
  the 363M budget). Cost too high for one-generation improvement.
- **Frac-connections.** CLOSED-NEGATIVE. Do not re-launch.
- **Uniform recurrent-depth (Huginn).** CLOSED-NEGATIVE. Do not re-launch.
- **Per-token MoR router.** Blocked on unbuilt sparse-gather kernel;
  Phase 3 concern.
- **SFT/DPO changes.** Post-pretrain concern. 500M v3 verdict says drop
  DPO or gate it hard. Flagship pipeline shape follows that.

---

## 6. Suggested pre-flagship execution order

**Before P4 seeds finish (~29h from now):**
1. Fix vocab encoder/decoder (§2.1) — verify by re-running `lm_eval` on
   arm B (5080, 15 min)
2. Skip `W_beta`/`W_alpha` under gdn2 (§4.1) — bit-verify state_dict
3. Fix spike aggregation dilution (§4.2) — retune target to 0.171

**After P4 seeds confirm 2-seed variance:**
4. Dead-code strip (§4.4) with byte-verify
5. Extend `brain_domain_transfer.py` (§4.5), collect plasticity baseline
6. Quadratic sparsity flag (§4.3) — opt-in only

**For flagship launch:**
7. Copy composed recipe → `configs/flagship_363m.yaml` with §3.2–3.8
   changes
8. VRAM/throughput probe at flagship shape with block_size 512 vs 768
9. **~5h shakedown** at flagship shape (~200M tokens) with watchdogs
10. Full flagship if shakedown clears

---

## 7. Open questions (for you)

**Q1.** Vocab fix — apply pre-flagship (breaks train/test mismatch), or
defer to post-flagship demo work? Pre-flagship is cleaner but changes
inference-time distribution for existing checkpoints.

**Q2.** Block_size 512 vs 768 for flagship — probe and decide, or lock
at 512 for one-fewer-variable comparability with W-arms?

**Q3.** How much do you want to invest in the vocab problem long-term?
Fresh BPE + retokenization would take ~40 min for build_bin, ~3h for the
train.bin regeneration on this corpus. Would give a cleaner vocab, unlock
larger-context evaluation with less token loss. But it's a full separate
project.

**Q4.** For flagship-launch VRAM budget — willing to sacrifice batch
back to 32 (grad_accum=4 at batch=8) if block_size 768 doesn't fit at
current batch 16?
