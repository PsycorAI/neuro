# v4 plan review: how to build a clearly superior spiking LM (2026-10-01, Neuro 48)

This reviews `docs/v4_recipe_research_2026_10_01.md` (the "current plan") in four iterations. It uses today's r15a/r15d results and a literature scan (2025-2026).
Status: a **proposal**. All builds and GPU runs need Glenn's OK. Every idea is also logged in `docs/to-do.html` (10-01 "v4 PLAN REVIEW" entry).

## TL;DR

1. **Today's data break layout C as written.**
   - r15a: removing the attention layers' MLP takes r14b from 50.14 to **63.77 (+27%)**.
   - r15d: a 2nd LIF layer in place of an attention+MLP layer gives **64.29**.

   So the per-token MLP carries most of the quality, and our LIF block (2d ternary neurons with a membrane) is a weak MLP. C would likely lose about 25%.
2. **Fix: make the spiking neuron layer a strong MLP.** The **dendritic spiking neuron layer (DSN)** models a two-compartment neuron:
   - the *apical* input decides whether the neuron fires (k-winner-take-all, i.e. lateral inhibition);
   - the *basal* input sets the size of its graded spike.

   It is the same math class as SwiGLU, but sparse, event-driven and inspectable one neuron at a time. With DSN, layouts B and C merge: **every block = route (sparse attention) + think (spiking neurons)**.
3. **What makes v4 distinctly superior is the memory system, not perplexity alone.** Transformers lack a brain-style memory. Ours combines:
   - a hippocampal Hebbian memory (layer 0; the *only* source of recall in every model we tested);
   - sink-free top-k attention over a sliding window.

   The result is **constant-memory streaming, recall far beyond the training window, and an auditable memory**. Add a measured interpretability scorecard and a neuromorphic stage 2. Perplexity must reach **parity** with a fair transformer at matched compute; the *wins* are memory, transparency and energy.
4. **Process: r16 becomes one full candidate (2 seeds) plus leave-one-out ablations, at 4 layers.** That is ~10 runs, one night on the 5080, and it replaces six single-flag screens that seed noise could not decide. Launch is ~10-05/06.
5. **Gaps found in the current plan:**
   - no data-quality step (the biggest benchmark lever);
   - width evidence from before the leak fix is over-trusted;
   - the queued 1B probe measures the obsolete layout;
   - r15b's SpikeGLU is sparse from step 0, which risks dead units. Fix it with a late k-ramp, as the LIF sparsity ramp did.

---

## 1. Evidence (30M rig = d 1024, ~70M params, 100M tokens; 1 seed unless noted; seed noise ~3-7%)

| Result | Number | Meaning |
|---|---|---|
| r14b [S, A, A] + canon + late ramp | **50.14** | best 3L |
| r14d [S, A, A, A] | **45.41**; haystack N8 100/56/34/24 | depth via attention+MLP pays; haystack weakens with depth |
| **r15a** attention layers without MLP | **63.77** (+27%); haystack better (N8 100/91/62/34) | the MLP carries most of the gain (57M vs 70M params; param-matched r15g pending) |
| **r15d** [S, A, S] + LayerScale 0.1 | **64.29** | the LIF layer is a weak MLP substitute; stacking it still fails |
| r13d neuron width 2048→4096 (post leak fix) | 61.2 → 59.9 (−2%) | width helps ppl only a little; the big pre-09-27 width curves were leak-era |
| r11c late sparsity ramp to 10% | +0.75% ppl | sparsity is nearly free when applied late |
| r12/r14c gap curriculum w0.15 | haystack solved, but +13-15% ppl and 2× time | the interference fix is too expensive at this dose |
| selectivity, spiking vs GELU | top-1 0.093 vs 0.128 | spiking ≠ interpretable unless designed in |
| signed multi-level spikes | 9.4% of 9.5% firing is ±1 | spikes are ternary in practice |
| attn_off / mem_off | recall unchanged / chance | **all recall is the Hebbian memory**: our unique asset |
| Engram v2 hash | keeps low bits, so tokens alias | must be fixed before any v4 screen |

Pending (by ~10-02 morning): r15c top-k16, r15b SpikeGLU, r15e fair 3L transformer, r15g no-MLP param-matched, r15f no-Engram.

## 2. Competitive landscape: what "superior" has to beat

| Model | What it is | Weakness we can beat |
|---|---|---|
| SpikingBrain 2.0 (5B, 2604.22575) | converted from Qwen3-4B (<7k A100 h); MoBA + sparse linear attention; INT8 spike coding; 64% sparsity on neuromorphic HW | not from scratch (a transformer in spiking clothes); no audit story; recall comes from attention |
| WTA Spikingformer (1B, 2604.11321) | softmax-free WTA spiking attention; 43.6 vs Qwen-1.5B 44.3 avg; ~17% energy | no associative memory; small claims |
| SpikeOPD (0.1-1.3B, 2608.27857) | spiking LMs *distilled* from transformers | dependent on a teacher |
| SymbolicLight (194M, 2605.21333) | LIF spike-gated dual path; ppl 7.7% *worse* than GPT-2; 4× slower | quality and speed |
| BDH (10M-1B, 2509.26507) | sparse positive ReLU neurons (n ≫ d), Hebbian state, monosemantic synapses, GPT-2 parity | not really spiking; GPT-2-level baselines; no exact episodic recall |
| HOLA (340M/15B, 2607.02303; not spiking) | delta-rule memory + small exact cache of "surprising" tokens; Wikitext 22.9 vs Transformer++ 26.9; RULER NIAH to 32k (16× train length) | an idea to *adopt*: it validates our hippocampus + exact-recall design |
| Gated DeltaNet-2 (2605.22791) | our memory rule | ours adds spikes, sparsity and audit on top |

**The gap we can own:** a spiking LM trained **from scratch** that has a **native Hebbian memory plus exact sparse attention**. It should match a **fair** transformer at matched compute on perplexity and **beat it** on memory beyond the window, transparency and energy, with every claim **measured**. No current system has all of these. Most spiking LMs are converted or distilled, and none ships an audit trail.

---

## 3. Iteration 1: review of the current plan

| # | Severity | Finding | Fix |
|---|---|---|---|
| F1 | **critical** | Layout C (12 LIF layers + 12 attention layers without MLP) is contradicted by r15a and r15d | Make the spiking layer a strong MLP (DSN); B and C merge |
| F2 | **high** | r15b's SpikeGLU uses k fixed from step 0, and gradients flow only to the winners, which invites dead units (the TopK-SAE problem). Our own LIF data say late sparsity beats constant sparsity. r15b may under-rate spiking MLPs | **k-ramp**: dense bilinear GLU until 60% of training, then anneal to target k (mirrors r11c); optional AuxK revival |
| F3 | high | No cheap haystack/interference fix (the gap curriculum costs 13-15%; r14d shows haystack weakening with depth) | Late-only gap dose in the candidate; memory at depth decided at G3; episodic cache in v4.x |
| F4 | high | **No data-quality step.** The corpus uses 3.3B of the 9.68B FineWeb-Edu tokens available. At a fixed 16B tokens, data quality moves benchmarks more than any architecture flag (FineWeb-Edu is worth ~4-6 MMLU points over raw CC) | Rebuild the mix toward FineWeb-Edu/DCLM-baseline in parallel (CPU/network). Space is fine: G: has 448 GB free and the WSL disk 566 GB (checked 10-01); the current corpus is 55 GB |
| F5 | high | Throughput: ~1B at 5.6-6.4 GFLOP/token needs ≥35% MFU for 16B tokens in 30 days, but v3 measured 27%. The queued probe measures FPT vs lif_exact_ml on the **obsolete draft shape** | Re-scope the probe to the final layout (batch, FP8, compile); keep GDN layers to 1-2 |
| F6 | medium | Width evidence from 09-11/09-23 predates the bf16 leak fix; post-fix width gave −2% ppl | Use width for recall and interpretability; don't budget perplexity on it |
| F7 | medium | r16 as six one-flag runs vs r14b at 1 seed: deltas under ~5% are undecidable, and interactions go untested | Full candidate (2 seeds) + leave-one-out ablations |
| F8 | medium | The Engram hash bug would confound every r16 result | Fix it first; all r16 runs use the fixed hash (one ablation measures it) |
| F9 | medium | Streaming is planned "later", but the attention design decides whether streaming is possible | **Sliding-window attention (window 512) now**: identical to full causal at T = 512, so it is free in training and gives constant-memory inference |
| F10 | medium | The M1 interpretability scorecard is a decision criterion but is not built | Build it during r16; score the r16 checkpoints before G3 |
| F11 | medium | No external anchor for a "beats transformers" claim | Fair in-house transformers at two scales plus published FineWeb/DCLM ablation curves (same data family) |
| F12 | low | src changes are uncommitted; the build-drift bug class (bitten 7×); post-training must accept the new modules | Commit after the r16 build; parity tests; `/audit-config` on the v4 config; smoke-test SFT/ORPO on a v4-shaped checkpoint |
| F13 | low | A 512-token context is short for 5-shot benchmarks and for long-context claims | Final long-context phase (~5% of tokens at 2k) with a banded top-k kernel, built while v4 trains |

## 4. Iteration 2: revised plan v2, and its review

**Plan v2.**
- Block 0 is a **hippocampal layer H**: today's S-layer with signed ternary LIF, canon, pre_norm, GDN-2 memory, gate_rank 16, mem_out_gate, LayerScale and linear atoms (no output norm).
- It is followed by L blocks of **A** (top-k, gated, sliding-window attention) → **DSN** (k-ramp).
- A 2nd H layer sits at ~2/3 depth.
- Engram is fixed (K = 65536).
- The data mix is rebuilt.

**Review of v2.**
- **R1. Is DSN really spiking?** Its neurons fire as events and carry a graded payload. Loihi 2 supports exactly this ("graded spikes"). But they have no membrane, and Glenn's preference for C was about dynamics in every neuron.
  - Add test **X2**: the DSN gate runs through a LIF neuron (`lif_exact_ml`, signed levels, learnable leak initialised small). This is the direct test of "every neuron is a LIF neuron".
  - Adopt it if it costs ≤2% ppl and ≤10% speed.
  - Otherwise the DSN neurons stay "fast-spiking" (no leak; real cortex has such cells) and membranes live in the hippocampal layers, where temporal integration is the point.
- **R2. The 2nd H layer is expensive.** It replaces one block's MLP and adds a launch-bound GDN loop.
  - Cheaper alternative, the **multi-reader hippocampus**: later layers read H0's memory with their own queries. That is two batched GEMMs from the saved chunk states, with no second write recurrence and ~1 day to build.
  - Decide at G3 by haystack-at-depth. The fallback (a 2nd H) needs no build.
- **R3. Throughput is still tight** (see §6.4).
  - FP8 is lossless on the activation side for the spike-input GEMMs, since ternary values are exact in FP8. The DSN and attention GEMMs take normal FP8 scaling.
  - The probe decides whether FP8 is worth it.
- **R4. Distinctiveness.** v2 risks looking like "a hybrid linear-attention transformer with sparse MLPs" (Qwen3-Next plus TurboSparse). Distinctiveness must be **visible in demos and numbers**, not just in the diagram. That leads to iteration 3.

## 5. Iteration 3: superior, not just competitive

Where can a spiking Hebbian model **beat** a transformer at the same budget?

1. **Memory beyond the window, in constant memory.** A transformer forgets everything outside its context, and its KV cache grows with length. Ours carries a synaptic memory. With sliding-window attention plus Hebbian memory, inference memory is constant and length is unbounded:
   - Samba-style hybrids trained at 4k pass passkey retrieval at 256k;
   - HOLA beats Transformer++ out to 16× the training length.

   **Claim to test:** haystack and RULER-NIAH at 2k-64k vs a fair transformer.
2. **Learning while reading, auditably.** Facts given in a stream can be recalled thousands of tokens later, and we can show the exact memory write that stored each one. No transformer can show this.
3. **Glass box, measured.** Scorecard numbers vs the fair transformer:
   - autointerp;
   - circuit size;
   - attention breadth (≤ k by construction);
   - memory traceability;
   - event-driven share.
4. **Energy.** Event-driven share and spikes per token, plus stage 2 conversion. Precedent: a MatMul-free LM on Loihi 2 ran ~3× faster at ~2× less energy than an edge GPU (2503.18002).
5. **Data efficiency.** Spiking wins at matched *tokens* (09-26). Show loss vs tokens, which matters as data runs out.

**Changes from iteration 3:**
- Add a **HOLA-style episodic cache** (v4.x, inference/long-context). Keep the tokens whose memory write residual (write strength × prediction error, which GDN-2 already computes) was largest in a bounded exact cache that attention can read. This is a hippocampal "index" of surprising moments: auditable, bounded, and free at 512 training.
- Add a **long-context final phase** (last ~5% of tokens at 2k windows) so the memory learns to carry state. Untrained carried state was +24% worse at gap 440 (09-29).
- Add **learn-while-reading** and **memory-traceability** evals to the scorecard.
- **Fairness rules for every claim:** transformer baselines get the same data, Engram (fixed), canon, independent windows, aux tasks and a tuned optimizer.

**Review of v3.**
- **Overreach risk.** Separate what must be in the *pretraining architecture* from what can be built *while v4 trains* (3-4 weeks).
  - Pretraining-critical: DSN, sliding-window gated top-k attention, H layer(s), the Engram fix, linear atoms, ternary spikes, the sparsity ramp, the aux/gap dose, data.
  - Built during training: streaming, audit recorder, scorecard runs, long-context kernel, episodic cache, stage 2.
- **Transfer risk** (30M → 1B; rankings can flip with depth). G3 must include haystack at depth and a fair transformer.

---

## 6. Iteration 4: the final plan

### 6.1 Architecture: "cortex + hippocampus"

```
tokens -> embedding + Engram (fixed hash, K 65,536, orders 2-3: editable lexical memory)
  H0   hippocampal spiking layer: signed ternary LIF neurons (n ~ 4d), canon taps,
       GDN-2 Hebbian memory (delta rule, logged writes), linear atoms + LayerScale
  x L  block:
       A   = sparse attention: top-k, sliding window 512, sigmoid output gate (no sinks),
             QK-norm, RoPE, GQA, canon taps
       DSN = dendritic spiking neurons: apical gate -> k-WTA (lateral inhibition),
             graded spike = apical x basal, n ~ 4d, late k-ramp to ~10% active, linear atoms
  memory at depth: multi-reader read of H0's memory, or a 2nd H at ~2/3 depth (chosen at G3)
  -> final norm -> LM head (+ MTP depth 2)
Inference: streaming with constant memory = LIF membranes + memory state + a 512-token KV ring.
```

The biological reading, for the pitch: **cortex** (wide, sparse spiking neurons with lateral inhibition), **hippocampus** (fast Hebbian memory, one-shot writes), **thalamic routing** (sparse attention picks ≤ k sources), **lexical memory** (Engram tables).

### 6.2 Recipe

| Component | v4 choice | Evidence | Status |
|---|---|---|---|
| Layer 0 | H0 = current S-layer recipe (lif_exact signed, canon 4, pre_norm, mem_out_gate 16, gate_rank 16, beta_floor 0.1, chunk 256) | r8-r14 | have |
| Spike levels | ternary (`lif_levels 1`) unless L4 says otherwise | levels 2-4 unused | r16 L4 |
| H output | no LayerNorm + LayerScale (linear atoms: each spike adds one fixed vector) | H1 half-right (r15d); interpretability | build b5, r16 L5 |
| MLP | DSN (k-WTA bilinear GLU, hidden ~4d, k-ramp to ~10%) | r15b + F2 fix; ReLU-sparse GLUs keep quality at 85-90% sparsity (TurboSparse, ProSparse) | build b3, r16 L3 |
| Neuron dynamics in DSN | LIF gate with learnable leak, **if** X2 ≤2% worse and ≤10% slower | Glenn's C goal | build b4, r16 X2 |
| Attention | top-k (k 16-64 per r15c), sliding window 512, sigmoid output gate, QK-norm, RoPE, GQA 16q/4kv | 2106.06899; gated attention 2505.06708; DeepSeek-V3.2 top-k (DSA) at scale; post-training can sparsify further (2512.05865) | r15c; build b2 |
| Canon in attention blocks | yes | canon_a −3% on S; Physics of LMs 4.1 | build b7, kept in C0 |
| Memory at depth | multi-reader or 2nd H | r14d haystack weakens with depth | G3 |
| Engram | high-bit hash, K 65,536, 4 heads, orders (2, 3) | bug fix; DeepSeek Engram 2601.07372 (+3-5 on MMLU/BBH/ARC-C) | build b1, r16 L1 |
| Sparsity | late ramp (λ 0.5 → 8 from 60%), ~10% firing | r11c | have |
| Recall aux | MQAR [4, 32] w 0.5 + **late-only** gap dose | r7/r12/r14c | build b6, r16 X1 |
| Optimizer | NorMuon 0.005 + AdamW, wd 0.1, head_wd 1.0, muon_batched_ns | v3 | have; route new matrices (gate-matrix bug class) |
| Aux heads | MTP depth 2 w 0.3; no z-loss | z-loss A/B null | have |
| Context | 512, then the last ~5% at 2k (banded top-k kernel) | F13 | build during the run |
| Data | rebuilt mix, heavier on FineWeb-Edu + DCLM-baseline, ~16B tokens | F4 | Glenn's OK + disk check |
| Precision | bf16; FP8 GEMMs if the probe shows ≥1.2× | Blackwell FP8; spikes exact in FP8 | G4 |

### 6.3 Width and depth (Glenn's question)

- **Neuron width** (neurons per spiking layer): n ≈ 4d for both DSN and H0.
  - More neurons at the same *active* count is the interpretability lever (MoE-X, BDH, Monet) and the recall lever (the memory layer).
  - Post-leak it buys little perplexity (r13d −2%), so don't widen beyond the speed budget.
  - "Neurons are cheap, firing is expensive" is the neuromorphic principle. On a GPU, wide-but-sparse only gets cheap with product-key routing, which is a v5 bet (§6.8).
- **Model width vs depth at ~1B:** d = 2048 with 12-14 blocks, a Llama-3.2-1B-like shape.
  - Wider d means bigger GEMMs and higher MFU on a launch-bound GPU.
  - Depth via attention+MLP keeps paying in our screens (3L→4L −11.5%), and 12-14 blocks is already deep.
  - Deep-thin (d 1536 × 20+) is better per parameter at sub-1B (MobileLLM) but ~1.3-1.5× slower for us (more launches). Not worth it at fixed wall-clock.

### 6.4 Size and throughput (analytic, ±15%; peak 104.8 TFLOPS bf16)

| Shape (d 2048, vocab 32k) | Total params | GFLOP/token | Days for 16B at 35% / 45% MFU |
|---|---|---|---|
| S1: H0 + 12 × [A + DSN 4d] | ~1.03B | ~5.6 | 28 / 22 |
| S2: H0 + 14 × [A + DSN 4d] | ~1.16B | ~6.4 | 32 / 25 |
| S1 + 2nd H (replacing one DSN) | ~1.07B | ~5.8 | 29 / 23 (+ GDN launch cost) |

At fixed compute, Chinchilla gives 0.85B on 16B tokens ≈ 1.0B on 13.6B (loss within 0.3%). So **param count barely matters at fixed compute**: pick the shape by measured tok/s, then set tokens to fit 30 days. The attention-heavy, GEMM-dominated layout should beat v3's 27% MFU, but G4 measures it.

### 6.5 r16, redesigned (30M rig at **4 layers**, 100M tokens, ~60-70 min per run on the 5080)

**Prerequisite builds** (CPU, default-off, bit-identical when off, parity-tested; ~half a day):

| Build | Change | Size |
|---|---|---|
| b1 | Engram high-bit hash (`engram_v2_hash: hi`) | ~5 lines |
| b2 | attention sigmoid output gate | ~10 lines |
| b3 | DSN k-ramp (dense → k from a start fraction) + optional AuxK | ~15 lines |
| b4 | DSN-LIF gate variant (`lif_exact_ml` on the apical input) | ~25 lines |
| b5 | `spk_out_norm: false` | ~5 lines |
| b6 | late-only gap dose (gap MQAR only after a set fraction) | ~5 lines |
| b7 | canon taps in attention blocks | ~10 lines |

**Runs.** Reference: r14d = 45.41 (same 4L shape, GELU MLPs, old hash).

| Run | Config | Question |
|---|---|---|
| C0 s0, s1 | [H, (A + DSN) × 3], all §6.2 choices | the candidate (2 seeds) |
| L1 | C0 with the old Engram hash | value of the fix (also informs the v3 story) |
| L2 | C0 without the attention output gate | is the gate earning its keep? |
| L3 | C0 with GELU MLPs instead of DSN | **price of spiking MLPs** |
| L4 | C0 with `lif_levels 4` | is ternary free? |
| L5 | C0 with the H output LayerNorm kept | are linear atoms free? |
| X1 | C0 + late-only gap dose | cheap haystack fix at depth |
| X2 | C0 with DSN-LIF gates | "every neuron is a LIF neuron" (Glenn's C) |
| T4 | fair 4L transformer (independent windows, fixed Engram, canon, same aux) | the honest margin |

That is 10 runs, ~11-12 h of compute (~16 h wall time with the 18:55-24:00 pause). Every run is also scored on M1.

**Decision rules.**
- **DSN goes into v4** if C0 is no more than ~3% worse than L3 and recall/haystack are equal. Otherwise G3 tries a wider DSN (6d at the same k). GELU MLPs in a minority of blocks are the last resort, because they weaken the story.
- **X2 is adopted** if it is ≤2% worse than C0 and ≤10% slower.
- **Gate, ternary, linear atoms, Engram fix:** keep each unless its ablation shows it *hurts* by more than ~1%. They are cheap and carry interpretability value.
- **X1 is adopted** if haystack N8 at gap 440 is ≥80% at a cost of ≤3% ppl.
- **From r15:** if top-k16 (r15c) costs more than 3%, C0 uses k = 32-64.
- **Overall:** C0 must beat r14d on the M1 scorecard and match it (within ~5%) on val_ppl.

### 6.6 Gates and timeline

| When | Step | GPU |
|---|---|---|
| 10-02 AM | r15 results; build b1-b7 (CPU) | 5080: r15 |
| 10-02 PM | Glenn's OK → queue r16 | 5080 |
| 10-02/03 | r16 runs. Meanwhile build M1 scorecard + memory audit mode. Data-mix rebuild (if approved) | 5080; CPU |
| 10-03 | 5090 free (post-training chain done) → **G4 1B layout probe**: tok/s, VRAM, batch, FP8 on/off, compile (~3-4 h) | 5090 |
| 10-03/04 | **G3 mid-scale**: ~150M, 10-12 layers, ~2B tokens; candidate vs fair transformer (transformer on the 5080), haystack at depth → memory-at-depth choice | 5090 + 5080 |
| 10-05 | Freeze the v4 config; `/audit-config`; commit; post-training smoke test on a v4-shaped checkpoint | |
| 10-05/06 | **Launch v4** (~26-30 days) | 5090 |

**Built while v4 trains** (weeks, no GPU contention):
- exact streaming (LIF `mem0`, canon/Engram history, KV ring) + a stream = window parity test;
- the audit "flight recorder";
- scorecard runs on checkpoints;
- the banded top-k kernel for the final 2k phase;
- the episodic cache;
- stage 2 conversion;
- a lean v4 model module (export plus parity vs the trained checkpoint) for code auditability.

### 6.7 Pitch scorecard: what we will claim, and how it is measured

| Claim | Measure | Against |
|---|---|---|
| Quality parity or better | val ppl; lm-eval (HellaSwag, ARC-e/c, PIQA, WinoGrande, LAMBADA, SciQ) | fair transformer at matched compute (70M and 150M); published FineWeb/DCLM ablation curves |
| Remembers beyond its window, in constant memory | haystack; RULER-NIAH 1k→64k; learn-while-reading stream test | the same transformer, with the same window and with full attention |
| Glass box | M1 scorecard; audit viewer page | the same transformer |
| Energy and neuromorphic fit | event-driven share; spikes per token; estimated energy per token after stage 2 | SpikingBrain 2.0, WTA Spikingformer (reported) |
| Data efficiency | loss vs tokens | the same transformer |

### 6.8 After v4: roadmap bets (logged in to-do, status TO-TEST/parked)

- **Ultra-wide neuron layers via product-key routing:** 64k-1M neurons per layer at constant compute (Memory Layers at Scale 2412.09764, UltraMemV2 2508.18756, Monet). This is the true "many neurons, few firing" design.
- **One shared neuron dictionary across depth:** a looped stack with k-WTA picking different neurons per pass. Looped models need sparse layers to scale (2605.09165), and BDH ties weights. One atlas for the whole model is very auditable. A cheap 30M screen could run on the 5080 *during* v4.
- **Uncertainty-aware memory writes:** Kalman Delta Networks (2609.07816) beat GDN at 750M-1.3B.
- **Spike-coded attention:** Q/K from neuron co-firing (BDH-style), so each attention edge is explained by shared active neurons.
- **Bigger Engram:** DeepSeek's U-law puts ~20-25% of sparse params in lookup memory; the compute cost is near zero.
- **Sleep consolidation:** replay the Hebbian memory into the weights offline (complementary learning systems).
- **Neuron-group MoE:** k-WTA at the group level, for more capacity at fixed compute (VRAM-limited on one 5090).
- **Parked:**
  - Transformer→SNN conversion: drops our memory and our uniqueness.
  - Online KD from Llama-3.2-3B: the teacher forward roughly doubles per-token compute on one GPU, and offline logits for 16B tokens are ~TBs.

## 7. Decisions needed from Glenn

1. **r16 redesign:** builds b1-b7 (default-off, parity-tested), then the 10 runs in §6.5.
2. **Build M1 scorecard + memory audit now** (CPU, during r16).
3. **Data upgrade:** rebuild the mix toward FineWeb-Edu/DCLM, after a G: free-space check (~1 day CPU/network).
4. **Size by compute, not by a param count:** "~1B" class, with final params (0.9-1.1B) and tokens set by the G4 probe to fit 30 days.
5. **B/C merge:** every block has a spiking neuron layer; X2 decides whether DSN neurons get membranes.
6. **5090 order after post-training:** G4 probe, then G3 mid-scale, then launch. Re-scope the queued neuron A/B probe into G4.

## 8. Risks

- **DSN quality** (r16 L3 decides). Fallbacks: wider DSN, or a k-ramp to a higher k (15-20%).
- **30M → 1B transfer.** G3 is the check. The recall-at-depth fallback (a 2nd H) needs no build.
- **Throughput under 6k tok/s.** Fallbacks: shape S1, FP8, or 13-14B tokens.
- **Interpretability may not beat the transformer by much at 1B.** The scorecard decides; claim nothing without numbers.
- **Streaming extrapolation untested** until the long-context phase and the streaming build. Keep the claims conditional until measured.

## Sources

- SpikingBrain 2.0: arXiv 2604.22575. SpikingBrain 1.0: 2509.05276.
- WTA Spiking Transformer: 2604.11321. SpikeOPD: 2608.27857. SymbolicLight V1: 2605.21333.
- Spike-driven LLM: 2604.16475. Event-driven LMs on neuromorphic HW: 2608.30439.
- BDH: 2509.26507.
- HOLA: 2607.02303. Gated DeltaNet-2: 2605.22791. Kalman Delta Networks: 2609.07816.
- Gated attention: 2505.06708. Top-k attention: 2106.06899. Sparse attention post-training: 2512.05865. DeepSeek-V3.2 (DSA).
- Samba (SWA + SSM extrapolation): 2406.07522.
- Engram (DeepSeek): 2601.07372.
- Memory Layers at Scale: 2412.09764. UltraMemV2: 2508.18756. Monet: 2412.04139. MoE-X: 2503.07639.
- Looped models need sparse layers: 2605.09165.
- TurboSparse; ProSparse; TopK SAEs (Gao et al. 2024); bilinear MLPs (Pearce et al. 2024).
- Neuromorphic LLM on Loihi 2: 2503.18002.
- FineWeb: 2406.17557. MobileLLM: 2402.14905. Physics of LMs 4.1 (canon layers).
- Intrinsic interpretability survey: 2604.16042.
