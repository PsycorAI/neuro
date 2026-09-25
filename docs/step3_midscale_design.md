# Step 3 — mid-scale reality check (design)

Date: 2026-08-02. Status: **agreed, Phase 0 launching.**
Prerequisite: Steps 1-2 CONFIRMED — see `docs/model_quality_redesign.md`
§"2026-08-02 — Step 1 + Step 2 CONFIRMED" (QK-norm fixed the attention seed
variance at 30M and 90M).

Step 3's job per the "path to ready to scale" plan: **confirm the seed
variance is actually gone — not merely smaller — at a scale closer to
target**, before committing GPU-weeks to Step 4 (the 100B+ token pretrain).

---

## 1. What the research changed

**Finding 1 — recall, not perplexity, determines the optimal attention ratio.**
A systematic study of hybrid linear-attention models
([arXiv 2507.06457](https://arxiv.org/html/2507.06457)) recommends a 3:1 to
6:1 linear:full ratio and finds that language-model loss "remains largely
flat across all ratio configurations" while recall "rises steadily as
full-attention layers are added and saturates around 3:1" — moving 24:1 → 3:1
nearly doubles RULER recall while shifting LM loss by less than 1.
Qwen3-Next and Qwen3.5 both ship 3:1 (three Gated DeltaNet blocks per
full-attention block).

**We currently run 6:1, and every screen this project has ever run measured
val_ppl only.** By the study's own result, val_ppl is exactly the metric that
*cannot* see this. That is a structural blind spot worth closing before 1B,
and it is why Phase 1 exists.

This also matches the project's own standing discipline: `mqar_cadence=2` was
rejected specifically because val_ppl improved while recall regressed
(see [[neuro-500m-v3-launched]]). Same failure mode, opposite direction.

**Finding 2 — QK-norm's documented benefit is learning-rate robustness.**
Wortsman et al. 2023 found QK-norm enabled stable training across three orders
of magnitude of learning rate. This de-risks the LR change that scaling width
requires (see §4).

**Counter-signal, recorded deliberately:** MiniMax's M2 abandoned the hybrid
design used in MiniMax-Text-01 and moved to full attention across all layers,
reporting that no hybrid variant reliably matched full-attention quality in
production. This does not change Step 3, but it is a real signal about where
this architecture family tops out and should be weighed before 1B.

---

## 2. Measured facts (not estimates)

Probes: `temp/probe_step3_throughput.py`, `temp/sweep_step3_shapes*.py`.
Methodology follows the project's `bench_blocksize.py` precedent — real
fwd+bwd steps with real optimizers, never a synthetic estimate.

**torch.compile works with `HybridAttentionBlock`** — never tested together
before. ~2.3x throughput AND lower peak VRAM. This is what makes Step 3 a
~2-day job instead of ~5. It is **not yet quality-validated** with the
attention block, hence the Phase 0 gate.

| Shape | Params | tok/s (5090) | peak VRAM |
|---|---|---|---|
| Step 3: d=1152, n=2752, L=8 | 267.1M | 17,690 @ batch 28 | 27.04 / 31.8 GB |
| " | " | 19,041 @ batch 32 | 28.68 GB |
| Ratio screen: d=640, n=1536, L=8 | 104.4M | 20,123 @ batch 32 | 18.86 GB |
| " (5080) | " | 9,923 @ batch 16 | 9.72 GB |
| Step 3 final: d=1152, n=2752, L=8 (block 512) | 276.5M | 17,690 @ batch 28 | 27.04 GB |
| " (block 640) | " | 16,193 @ batch 24 | 30.01 GB (WDDM ceiling) |
| " (block 640, batch 28) | " | 4,851 @ batch 28 | 34.62 GB (spill) |

**WDDM spill reconfirmed live**: on the 5080, batch 32 needed 19.02 GB on a
16 GB card and throughput **collapsed 5,803 → 1,367 tok/s** — the same
failure mode as [[neuro-500m-v3-vram-sweetspot]], caught here in a probe
rather than in a live run. The batch ceilings above are measured.

---

## 3. Confound found while designing (must correct in analysis)

`src/model.py:1259`:
```python
spike_rate = torch.stack([s.mean() for s in all_spikes]).mean()
```
This is a **flat mean over all layers, and attention layers emit exactly zero
spikes** (`hybrid_attn.py` returns a zero spike stub by design). Therefore:

- 1 attention layer of 8 → reported spike rate = 0.875 × true spiking rate
- 2 attention layers of 8 → reported spike rate = 0.750 × true spiking rate

Going from 1→2 attention layers **mechanically lowers reported spike rate by
~14% with identical spiking layers**, and weakens the sparsity penalty by the
same factor. **Comparing the Phase 1 arms on raw spike rate would falsely
conclude the 2-attention arm is sparser.** Normalize by `n_spiking/n_total`
before comparing. Flagged here because the same artifact will recur in every
future attention-ratio comparison.

---

## 4. The plan

### Phase 0 — compile parity gate (~1.5h)

Rerun 30M seed 0 with `compile: true`, everything else identical to
`configs/screen_qknorm_30m_s0.yaml`, and compare to that run's known eager
result of **val_ppl 140.30**.

Justification: the entire Phase 1/2 time budget rests on compile being
quality-neutral, and this project has a real silent-miscompile history — the
`lif_parallel` data-dependent FPT loop was miscompiled into a spike collapse
(0.185 → 0.11) and ~2x worse val_ppl before being caught
([[reference-torch-compile-delta]]). Compile has been validated on the
pre-attention architecture only.

**Gate:** val_ppl within ~2% of 140.30 and spike rate within noise → proceed.
Otherwise drop compile and re-budget Phases 1-2 at ~2.3x the wall clock.

### Phase 1 — attention-ratio screen (~8h)

| | Arm A | Arm B |
|---|---|---|
| attention_layer_positions | `[3]` | `[3, 6]` |
| linear:full | 7:1 | 3:1 |
| params | 107.3M | 104.4M |

Shape d=640 / n_neurons=1536 / d_mem=640 / L=8 — a 90M-class model at the
**target aspect ratio** (n≈2.4d) and target depth, so both ratios are
expressible and the arms are near param-matched (2.8% apart; an attention
block is *smaller* than a spiking block at this aspect ratio).

2 seeds per arm, 150M tokens each (~1.4 tok/param, matching the 90M screens'
1.3). **All four runs on the 5090, same batch** — the project already lost
time to a cross-GPU/batch confound during the seed-variance investigation.

**Primary metrics: `recall_eval` (with `--replay split`) and memory-Δ. NOT
val_ppl** — per Finding 1, val_ppl is expected to be flat and is not the
decision metric.

**Decision rule:**
- 3:1 lifts recall AND memory-Δ holds → adopt 3:1. Attention and Hebbian
  memory are complementary.
- 3:1 lifts recall BUT memory-Δ drops sharply → attention is cannibalizing
  the Hebbian pathway. **Escalate — this is a quality-vs-differentiator
  judgement call for Glenn, not a metric threshold.**
- No recall difference → keep 7:1 (fewer attention layers preserves the
  sparsity/energy claim and the differentiator at no measured cost).

### Phase 1 result (2026-08-03) — 7:1 wins decisively on recall

All four runs completed. Full log: `logs/recall_phase1_full.log`.

| Arm | Attn | val_ppl (final) | mem-Δ | Recall N=4/8/16/32/64 (--replay split, B=64) |
|---|---|---|---|---|
| A (7:1) s0 | [3] | 48.56 | +231 | **100/95/78/75/59%** |
| A (7:1) s1 | [3] | 47.06 | +243 | **100/98/94/80/47%** |
| B (3:1) s0 | [3, 6] | 47.84 | +255 | **31/22/9/6/3%** (~chance) |
| B (3:1) s1 | [3, 6] | *(pending)* | *(pending)* | **52/22/11/5/3%** (~chance) |

**val_ppl and memory-Δ said the arms were tied / B slightly ahead** — but
recall collapsed to chance on both arm-B seeds, the exact
`mqar_cadence=2`-class failure mode this project's discipline exists to catch.
A without-`--replay split` control gave the same qualitative answer (arm A:
97/89/28% vs arm B: 41/8/2% at N=4/16/64), so it's not a split-replay
artifact — arm B just failed to learn associative recall at all.

**Why the opposite of arXiv 2507.06457's direction:** their "more attention =
more recall" was measured on RULER-style **in-context** tasks; our metric
(`--replay split`) is **persistent-memory** recall, which IS the project's
differentiator. On persistent-memory recall, more attention appears to
compete for the MQAR-aug training signal without teaching the Hebbian
pathway to store keys.

**Standing convention updated:** memory-Δ alone is insufficient to judge the
memory pathway — arm B had HIGHER memory-Δ (+255 vs +231) while having
~zero recall. Always pair memory-Δ with recall_eval.

Details in memory `neuro_phase1_attnratio_finding.md`.

### Phase 2 — Step 3 proper (~52h)

3 seeds (0/1/2, for continuity with all prior data), sequential on the 5090.

| Field | Value | Rationale |
|---|---|---|
| `d` / `n_neurons` / `d_mem` | 1152 / 2752 / 1152 | **n≈2.4d matches 500M v3 and the 1B design. Every prior 30M/90M screen used n=d and was never shape-faithful to the target.** |
| `n_layers` | 8 | matches the 1B design's depth |
| `attention_layer_positions` | from Phase 1 | — |
| `attention_heads` | 9 | head_dim 128, the modern convention |
| params | 267.1M | mid-band of the 200-350M target |
| `max_tokens` | 1.1B | 4 tok/param — 3x the relative budget of the 90M screen |
| `batch_size` / `block_size` | 28 / 512 | 27.04 GB leaves 4.8 GB headroom for a 2-day unattended run |
| `lr` | 5.0e-4 | log-interpolated between this project's own points: 6e-4 @ d=640, 4e-4 @ d=2048 |
| `sparsity_lambda` / target | 0.3 / 0.15 | unchanged from the QK-norm screens |
| `compile` | true (post-gate) | mode `default`, never `reduce-overhead` |

**Win condition:** the 3 seeds converge within noise (comparable to the ~8-11%
spread seen at 30M/90M post-QK-norm), with no sign of the spread *growing*
over the trajectory.

**5080 during Phase 2:** runs the λ=0.5 multi-seed confirm — the remaining
loose end from the overnight sparsity sweep — per the standing
"don't leave the 5080 idle" preference. Independent of the 5090 work.

---

## 5. Deliberate non-goals

Step 3 answers **one** question well: *is the seed variance gone at 267M?*
It deliberately does NOT validate at scale:

- sparsity λ (held at 0.3 for a clean comparison; λ=0.5 screened separately)
- `block_size` 640 (the 500M v3 quality lever — measured here as ~free on
  throughput, but changing it would add a variable to the scale comparison)
- curriculum warmup (off, as in the QK-norm screens)
- Engram-v2 K=16384 (the 1B design flags this as its own screen)

Each of those is a real open item; bundling them into Step 3 would confound
the one result that is currently blocking the project.

**Update 2026-08-03 — block_size 640 probed, deferred deliberately.** At the
Step 3 shape, block 640 sits right at the 5090's WDDM ceiling: batch 24 =
30.01 GB (only 1.8 GB margin, high spill risk for an unattended 2-day run);
batch 28 already spills and cratered throughput to 4,851 tok/s. Block 512 at
batch 28 is 8% faster (17,690 tok/s) AND leaves 4.8 GB headroom, so keeping
block 512 for Step 3 is both cleaner (one fewer variable vs Phase 1) and
safer. Block 640 stays a real quality lever for Step 4 / the 1B design,
just not here.
