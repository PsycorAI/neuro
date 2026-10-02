# Interpretable, spiking, and better: architecture rethink (2026-10-01)

Glenn's goals: the model must be **interpretable and auditable** (safety/alignment), stay **spiking and close to neuromorphic** (our differentiation from transformers), and still **beat transformers, BDH and other spiking LMs on most metrics**.
Status of every idea below: **TO-TEST** (nothing launched). GPU runs need Glenn's OK.

## 1. Short answers

1. **Losing interpretability matters, and our current best stack (r14b) has lost much of it.** Two thirds of its layers are a standard transformer layer: dense softmax attention plus a dense 4096-unit GELU MLP.
2. **No, we don't have to drop attention.** Attention is the *most* inspectable part of a transformer: it records which earlier tokens were read. The hard-to-interpret parts are dense MLP neurons and the dense residual stream. Two recent results back this up:
   - The neuromorphic literature treats sparse, winner-take-all attention as neuromorphic: lateral inhibition across time ([2604.11321], [2601.00245]).
   - Top-k attention keeps transformer quality with k ≈ 3% of keys [2106.06899], and post-training can cut attention to 0.4% of its edges at no loss, up to 7B [2512.05865].

   So the rule is: **keep attention, but make it sparse and spike-driven, and remove the dense MLPs.**
3. **Agreed: a mostly-transformer stack weakens our story.** The plan below makes every nonlinear feature a spiking neuron again.
4. **Yes, there is a credible path.** Our own data say the spiking memory layer is a *better* first layer than attention. The things that hurt are fixable or untested (sections 2 and 4).

## 2. What our results say (30M, 100M tokens)

| Evidence | Numbers | Meaning |
|---|---|---|
| Spiking layer beats attention as layer 0 | [spk, attn] 61.4 vs 2L tfmr 68.8; [spk, attn, attn] 51.3 vs 3L tfmr 59.4 | The spiking front end (LIF + GDN-2 memory + Engram) is an asset. Confounded, see "Fairness" below. |
| Attention layers do NOT do recall | r13a attn_off haystack N8 100/89/64 vs full 100/80/54; mem_off = chance | All recall is the spiking layer's memory. Attention buys perplexity only. |
| A 2nd spiking layer hurts | [spk, attn, spk] 63.4/64.7 vs [spk, attn] 61.4 | Pathological: adding a layer should not hurt. See H1 below. |
| Width helps | n 2048→4096 (param-matched): 61.2 → 59.9 | Also the recall lever, and the interpretability lever (SAE / MoE-X / BDH). |
| The pure-spiking gap is local context | ratio vs tfmr 1.72 at positions 4–15 and 1.64 at 16–63 | Attention fixes precise recent-context use. |
| Our "attention" layer is two thirds dense MLP | 4.2M attention + 8.4M GELU MLP params per layer | r13a's gain is unattributed: attention, MLP, or both. |
| **Spiking ≠ interpretable by default** (new, `temp/diag_neuron_selectivity.py`) | Layer-0 units at matched activity: top-1 token share 0.093 (spiking) vs 0.128 (tfmr GELU); p90 0.235 vs 0.434; token-entropy score 0.306 vs 0.306 | Our 2048 neurons are no more selective than dense MLP units. This is a crude proxy on 4k tokens, but interpretability must be **designed in**, not assumed. |

**H1 (new, likely root cause of "depth hurts spiking").** A spiking block returns `LayerNorm(r + ff)` and the loop adds it to the residual. Every spiking layer therefore injects a unit-norm vector and cannot start near-identity, unlike pre-norm attention blocks. r13b's `depth_scaled_init` scaled `W_ff`, but the LayerNorm undoes that scaling, so **r13b was a no-op test, not evidence**. The slow start (eval-1 180–190 vs 153) fits this. The fix is cheap: drop the output norm, or add a LayerScale/ReZero gain initialised small.

**Fairness of the "beats transformer" claim (must fix before external claims).**
- No transformer baseline used `data_independent_windows`, which was worth about 3% on spiking (09-25).
- None had Engram v2.
- They also used a different optimizer (AdamW at 1e-3, which was best for them).
- The true margin is probably smaller than the 14–16% we quote.

## 3. Target design: "glass-box spiking LM"

Every component is one of four auditable, spike-driven primitives:

| Primitive | Role | Audit trail | Neuromorphic analogue |
|---|---|---|---|
| **Wide spiking neuron layer** (signed LIF, lateral inhibition, n ≫ d) | All nonlinear feature computation | Which neurons fired (spike raster); per-neuron atlas | Cortex: LIF + k-winner-take-all |
| **Associative memory** (GDN-2 delta rule, already ours) | Recall and long-range context | Write log (token, strength, neurons) and read attribution | Hippocampus: fast Hebbian synapses |
| **Spiking top-k attention** (Q/K from spikes, at most k sources per token, no dense softmax over all keys) | Precise local context (the 4–63 gap) | Exact list of at most k tokens read per token | Lateral inhibition across time |
| **Engram n-gram tables** (already ours) | Cheap lexical knowledge | Which table entries were used; entries are editable | Lookup memory |

The layer pattern goal is [S, A, S, A, S] (spiking layers carry all nonlinearity, attention only routes), if H1 is fixed. The fallback is [S, A+SpikeGLU, A+SpikeGLU].

## 4. Ideas, ranked

### Tier 0: measurement (needed for every claim)

| # | Idea | Cost | Success criterion |
|---|---|---|---|
| M1 | **Interpretability scorecard** as a standard eval, applied to r14b, tfmr3, BDH and every candidate. Five measures: (a) autointerp, where local Llama-3.2-3B explains each unit's top examples and then predicts held-out activations (Bills et al. 2023 / delphi style); (b) a label-free coherence score for binary units (TMS, [2607.17770]); (c) circuit size, the number of units needed for 90% of the logit on MQAR and simple grammar probes; (d) attention breadth, the tokens needed for 90% of attention mass; (e) memory audit, the share of correct recalls traceable to one logged write | ~1 day build, cheap to run | One table per model, so "interpretable" becomes a number we track like val_ppl |
| M2 | **Honest baselines**: tfmr3 with independent windows (± Engram, same MQAR aux); r14b with Engram off | config only, 2–3 runs | The real margin vs transformers |
| M3 | **BDH-GPU on our data** at ~30–74M params, same evals (pathwaycom/bdh code) | half-day integration plus 1 run | Direct BDH comparison on ppl, recall and interpretability |

### Tier 1: diagnose and unlock (cheap flags, about 50 min each)

| # | Idea | Code | Decides |
|---|---|---|---|
| D1 | r14b with **attention layers without their MLP** (`attn_ffn: false`), optionally param-matched by width | ~5 lines | Is r13a's gain attention or the dense MLP? |
| D2 | **Spiking-depth unlock**: [spk, attn, spk] with `spk_out_norm: false` plus LayerScale initialised to 0.1 (also try it on r14b's single S-layer) | ~10 lines | Can spiking layers stack like transformer layers (H1)? |
| D3 | A late spiking layer as a per-token spiking MLP: β=0 (no leak) and no memory | ~15 lines | H2/H3: does a 2nd memory interfere, or does membrane smoothing blur late features? |
| D4 | **Top-k attention** in r14b's attention layers, k = 16 of 512 (softmax over survivors) | ~15 lines | Is auditable attention free? The literature says yes. |

### Tier 2: put spikes back into every nonlinearity (moderate code)

| # | Idea | Why it could keep quality |
|---|---|---|
| S1 | **SpikeGLU** MLP replacing the GELU MLP: h = spike(W1·x) ⊙ (W3·x), with a signed threshold spike (β=0) gating a graded value (Loihi-2 style graded spike) and ~5–10% firing | Keeps a SwiGLU-like gate (BDH uses the same x⊙y sparse gating). Sparse units can be interpreted. |
| S2 | **Spike-coded attention**: Q/K from the spiking layer's ternary spikes (co-firing overlap + RoPE), top-k WTA | BDH's attention is neuron-space linear attention, so each attention edge is explained by shared active neurons |
| S3 | **Wide spiking layer, same active budget**: n 2048 → 8192, firing ramped to ~2.5% (≈ the same ~200 active neurons), k/v/q computed from the decoded vector instead of from raw spikes so params stay sane (`mem_from_decoded`) | Width is our best quality and recall lever (r13d, 09-23 probe) and the main monosemanticity lever (SAE, MoE-X, BDH n=32k) |
| S4 | **Neuron-keyed memory**: memory keys/queries = normalised spike vectors, so the state is (d_mem × n) and each column is "what neuron i is associated with" (BDH-style readable synapses; local Hebbian update) | Auditable memory. Recall risk: screen after S3. |

### Tier 3: neuromorphic quality levers (cheap)

| # | Idea | Note |
|---|---|---|
| Q1 | Longer **canon** taps (canon_a 8 / 16), i.e. learnable synaptic delays | canon_a 4 was −3%; the local-context gap spans 4–63 tokens |
| Q2 | **Adaptive-threshold (homeostatic) LIF** instead of the sparsity penalty | SpikingBrain / ALIF; biologically grounded rate control |
| Q3 | Multi-timescale / dendritic neurons | PLIF was noise-level at 2L; low priority |
| Q4 | **Weight-tied spiking stack** (BDH-style): one neuron dictionary reused across depth | Only after D2 works. A single dictionary is much easier to audit. |

### Tier 4: safety demos (after the architecture settles)
- **Unlearning by ablation**: silence the neurons or table entries for a concept and show it disappears (Monet-style expert removal).
- **Audit viewer**: per token, show the spike raster, memory writes and reads, attention sources and Engram hits, as an HTML page.

## 5. Proposed first round (r15, 30M, 100M tokens, about 6 runs ≈ 5 h on the 5080)

| Run | Change vs r14b (50.14) | Needs |
|---|---|---|
| r15a | D1: attention layers without MLP | flag |
| r15b | S1: SpikeGLU instead of the GELU MLP | code |
| r15c | D4: top-k attention, k=16 | flag |
| r15d | D2: [spk, attn, spk] with output-norm fix (vs 63.4/64.7) | flag |
| r15e | M2: tfmr3 with independent windows (+ Engram) | config |
| r15f | M2: r14b with Engram off | config |

**Decision rule.**
- If r15a keeps most of the gain and r15d stacks: go [S, A(top-k, no MLP), S, A, S], so every feature is a spiking neuron.
- If r15a loses but r15b holds: go [S, A+SpikeGLU, A+SpikeGLU].
- In both cases, M1 scores decide the final layout. Any candidate must stay within about 5% of r14b's val_ppl, keep recall at 100%, keep haystack, and **beat r14b on the M1 scorecard**.

## Sources
- BDH [2509.26507]: neuron dim n ≫ d, ReLU sparse positive (~5%), linear attention in neuron space, (3+o(1))·n·d params, monosemantic synapses, matches GPT-2 at 10M–1B.
- Weight-sparse transformers [2511.13653]: sparsity trades capability for interpretability; bigger models improve the frontier.
- Sparse attention post-training [2512.05865]: 0.4% of edges, same loss, up to 7B.
- Top-k attention [2106.06899].
- MoE-X [2503.07639]: a wide sparse ReLU MLP beats GPT-2 ppl with interpretability above SAEs.
- Monet [2412.04139]: monosemantic experts; unlearning by expert removal.
- WTA spiking transformer [2604.11321].
- Modern neuromorphic AI survey [2601.00245].
- Neuromorphic spike-based LLM (Natl Sci Rev 2025).
- TTFS spiking LLM 1.5B [2609.05151].
- TMS monosemanticity score [2607.17770].
