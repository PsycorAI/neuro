# v4 recipe: research and iteration (2026-10-01)

Inputs:
- `docs/interpretable_spiking_plan_2026_10_01.md` (goals and ideas)
- `docs/code_review_interpretability_2026_10_01.md` (review findings)
- the 30M screens r5–r14 (round 15 queued)

Status: a **proposal to iterate on, not a final recipe**. Final choices wait for the r15 and r16 results and a mid-scale check.

## 0. Targets and constraints

| Target | Detail |
|---|---|
| Size and time | ~1B params, ~16B tokens, under 30 days on the 5090, i.e. ≥ 6.2k tok/s sustained |
| Must be | interpretable and auditable; spiking and close to neuromorphic; better than a fair transformer, BDH and published spiking LMs on most metrics |
| Throughput fact | Every extra spiking layer *with memory* costs a lot of wall time: the GDN-2 chunk loop is launch-bound (r13b 80 min vs r13a 49 min at 30M). Attention layers are nearly free (flash). |

## 1. The current v4 draft is obsolete

`configs/flagship_v4_363m.draft.yaml`:
- has 8 spiking layers and 1 attention layer (the layout our depth screens say is worst);
- uses FPT neurons (never converge), so it has no `lif_exact`;
- has no signed spikes, canon, pre_norm, output gate or late sparsity ramp, and a sparsity target of 0.15;
- carries the Engram hash bug (K = 8192 with a 32k vocab, a 4-way merge of every token).

v4 must be rebuilt from the screened recipe.

## 2. What the evidence implies

1. **The spiking memory layer is a better first layer than attention, and it is the only thing that does recall.** Keep it at layer 0. Attention adds no recall (attn_off is unchanged).
2. **Attention plus an MLP per layer is what buys language-modeling quality.** r13a [S, A, A] = 51.3 (2 seeds) vs [S, A, S] = 63–65. Which half matters (attention or MLP) is answered by **r15a/r15g**.
3. **Spiking layers do not stack yet.** The likely cause is the unit-scale LayerNorm output (H1); **r15d** tests the fix. If stacking works, spiking layers can be the MLPs of a deep stack.
4. **Width helps everything:** quality (r13d), recall (09-23 width probe) and, per the literature, interpretability. Use n ≈ 4d neurons per spiking layer at ~2.5–5% firing.
5. **The late sparsity ramp is nearly free:** 10% firing for +0.75%, 5% for +2%.
6. **The haystack gap curriculum works but is too expensive** at the current dose: +13–15% ppl and 2× step time on signed stacks.
7. **Our measured lead over transformers is inflated** by baseline unfairness (r15e will tell).
8. **r14d (added 14:40): depth via attention keeps paying, but haystack weakens.** 4L [S, A, A, A] = 45.41, vs 51.3 at 3L (+17% params, −16% tok/s). Haystack drops to N8 56/34/24 even though memory retention is fine (half-life 134/75). With more attention layers, the memory's read is used less under interference. **v4 implication:** place memory layers at more than one depth (layer 0 plus a mid or late layer) so the memory read enters near where it is used, and test the cheap gap dose at depth.

## 3. The v4 shape: a transformer-shaped stack where every MLP is a spiking neuron layer

The cleanest unification of the options:
- **Every layer's nonlinearity is a spiking neuron layer:**
  - either the existing LIF path (d → n LIF → n → d), "S";
  - or SpikeGLU (k-WTA graded spikes), "SG".
- **Every token-mixing step is top-k, gated attention** ("A"): auditable, sink-free.
- **A few layers also carry the GDN-2 associative memory** ("S+mem"): recall and long range, with logged writes.
- **Engram n-gram tables** (hash fixed) at the input.

Candidate layouts at d = 2048, vocab 32768, T = 512 (analytic estimate; peak taken as 104.8 TFLOPS, `scratchpad v4_estimate.py`):

| Layout | Params | GFLOP/tok | Days for 16B at 30% / 40% MFU | Q risk | I | N |
|---|---|---|---|---|---|---|
| A ref: [S+mem, A(dense GELU) × 15] (r14b scaled) | 0.97B | 5.4 | 31.7 / 23.8 | lowest | poor (dense MLPs) | poor (~3% events) |
| **B: 3 × S+mem (n = 8192) + 13 × [A top-k gated + SpikeGLU]**, 16L | 1.05B | 5.9 | 34.8 / 26.1 | r15b/r15c decide | good | medium (no membrane in SG) |
| B′: as B, but memory k/v/q from the decoded vector | 0.94B | 5.2 | 30.8 / 23.1 | +1 unknown | good | medium |
| **C: 12 × S (n = 8192, memory in 3) alternating with 12 × A (top-k gated, no MLP)**, 24L | 0.90B | 5.1 | 30.0 / 22.5 | r15d/r15g decide | good | **best** (LIF dynamics everywhere) |
| C′: 10 + 10, memory in 2, 20L | 0.75B | 4.2 | 24.5 / 18.4 | as C | good | best |

Reading:
- **C is the most spiking and neuromorphic design**: every nonlinearity is a LIF neuron with a membrane, and attention only routes.
- **B is the safer quality bet** if SpikeGLU ≈ GELU.

All candidates sit near the 30-day line, so the speed levers in §6 matter. Measured v3 MFU was 27% (all-spiking, launch-bound). The attention-heavy layouts should do better, but this must be **measured with a 1B probe**.

## 4. Recipe components (status)

| Component | Proposal | Status / evidence |
|---|---|---|
| Neuron | `lif_exact`, signed; `lif_levels` 1 (ternary) if r16 confirms, else 4 | levels 2–4 are ~unused (9.4% of 9.5% firing is ±1) |
| Sparsity | late ramp (λ 0.5 → 8 from 60%), target 5–10% by budget | r11b/c |
| Spiking-layer hygiene | pre_norm, mem_out_gate 16, canon_a 4, gate_rank 16, **LayerScale or no output norm** | r15d; spk_out_norm TO-BUILD |
| Width | n ≈ 4d | r13d, 09-23 probe; interpretability literature |
| Memory | GDN-2 chunked (chunk 256, gate_norm, no chunk ckpt), beta_floor 0.1; 2–3 memory layers, one at layer 0 | v3 recipe |
| Attention | **top-k (k ≈ 16–32)**, **gated output (sigmoid, G1)**, QK-norm, RoPE, GQA 16q/4kv; maybe canon taps | r15c; gate and canon TO-TEST (r16) |
| MLP | SpikeGLU (k-WTA) or none (S-layers are the MLPs) | r15a/b/g |
| Engram v2 | **high-bit hash**, K = 65536, 4 heads, orders (2, 3) | BUG fix TO-BUILD, then screen |
| Recall aux | MQAR [4, 32] at w 0.5 (settled) + **cheaper haystack dose**: late-only gap or lower weight | TO-TEST (r16) |
| Optimizer | NorMuon 0.005 + AdamW, wd 0.1, head_weight_decay 1.0; maybe muon_attn_lr | v3; attention LR untested at scale |
| Aux heads | keep MTP (depth 2, w 0.3); drop z-loss unless an A/B supports it | z-loss A/B was null/negative |
| Context | block 512 (bs1024 was worse in a leak-era, attention-free screen); reconsider for attention-heavy v4 with a short long-context stage at the end | caveat |
| Data | 16B tokens from the 14.5B corpus (~1.1 epochs) or curated | v4 plan |

## 5. Interpretability and audit features (build before v4 post-training)

1. **Exact streaming**: LIF `mem0`, canon and Engram history, stream = window parity test. This is needed for neuromorphic deployment and for faithful audit trails.
2. **Audit hooks**, at about 1 KB per token: spike indices per layer, top-k attention sources, memory write / erase / decay plus read attribution, and Engram buckets. This is a "flight recorder" for every generation.
3. **Interpretability scorecard (M1)**, run on every candidate and on v4 checkpoints:
   - autointerp (local Llama-3.2-3B);
   - TMS coherence;
   - circuit size;
   - attention breadth;
   - memory traceability;
   - event-driven share.
4. **Safety demos** after v4: unlearning by neuron or table ablation; an audit viewer page.

## 6. Speed levers (to fit under 30 days)

- **Fewer memory layers** (2–3), since they dominate launches.
- **Fuse the 7–8 spike-input GEMMs** into one (exact).
- **Compile** with per-block CUDA graphs for the GDN chunk loop (docs/cuda_graphs_v4_design.md; gated on GPU-busy ≤ 60%).
- **Batch** is the lever (launch-bound).
- **Top-k attention:** explicit scores are affordable at T = 512 (attention ≈ 6% of FLOPs). Write a fused kernel only if the probe shows it is needed.
- **The 1B probe** (queued on the 5090, neuron A/B) plus a new layout probe give the real tok/s.

## 7. Neuromorphic stage 2 (after pretraining, optional)

Convert the dense inputs to spike-driven ones with a short quantization-aware phase:
- residual → signed integer spikes before every projection (the "membrane shortcut");
- integer-trained, binary at inference (cf. arXiv 2604.16475).

This raises the event-driven share from ~3% toward a majority without risking pretraining quality, and the quality cost is measured at the end. It mirrors how BitNet-style activation quantization is done.

## 8. Decision gates and timeline

| Gate | When | Decides |
|---|---|---|
| G1: r15 results | 10-02 | B vs C; MLP or none; top-k OK; stacking OK; fair-baseline margin; Engram value |
| G2: r16 | 10-02/03 | **independent of r15:** Engram hash fix, ternary spikes, gated attention, attention canon, late-gap haystack; then the combined candidate × 2 seeds + scorecard + fair transformer (+ canon) + BDH |
| G3: mid-scale | ~10-04/05 | ~150M / 3B tokens on the 5090 (~1–1.5 days): final candidate vs fair transformer at the same budget. Checks that the 30M ranking transfers. |
| G4: 1B probe | ~10-05 | tok/s, VRAM, batch at the final layout; set tokens to fit 30 days |
| G5: launch v4 | ~10-06 | with streaming and audit hooks ready before post-training |

## 9. Proposed r16 (small flags, all default off; build needs OK)

| Run | Change vs r14b | Build |
|---|---|---|
| r16a | Engram high-bit hash | flag (~5 lines) |
| r16b | ternary spikes (`lif_levels: 1`) | config |
| r16c | gated attention output (sigmoid G1) | flag (~10 lines) |
| r16d | canon taps in attention blocks | flag (~10 lines) |
| r16e | late-only gap curriculum (gap MQAR in the last 30%, w 0.15) | flag (~5 lines) |
| r16f | spiking output without LayerNorm + LayerScale (if r15d is positive) | flag (~5 lines) |

Then the combined candidate (per G1) × 2 seeds.

## 10. Risks
- The 30M → 1B transfer is unproven, hence G3.
- SpikeGLU or top-k may cost more than the literature suggests at our scale; r15 measures it.
- Throughput may come in under 6.2k tok/s. Fallbacks: 0.75–0.9B params (C′), or 12–14B tokens.
- Interpretability may not improve even with width and k-WTA. The scorecard decides; don't claim it without numbers.

## Sources
- Gated attention (Qwen, NeurIPS 2025): arXiv 2505.06708.
- Hybrid linear attention, systematic analysis: arXiv 2507.06457. LM quality is flat across ratios; recall saturates near 3:1. Here recall comes from the memory instead.
- Spike-driven Transformer: arXiv 2307.01694 (membrane shortcut).
- Spike-driven LLM: arXiv 2604.16475 (integer → binary spikes).
- Top-k attention: arXiv 2106.06899.
- Sparse attention post-training: arXiv 2512.05865.
- MoE-X: arXiv 2503.07639.
- BDH: arXiv 2509.26507.
