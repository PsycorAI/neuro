# Closing the honest val_ppl gap — research plan (2026-09-27)

Goal (Glenn): 30M honest val_ppl ratio spiking / transformer from **1.45× → ≤1.0×, ideally ≤0.9×**, keeping the recall and sparsity advantages.
Control: `screen_honest_lifexact_s0_30m` (99.3, recall 100/100/94). Baseline: `screen_tfmrfix_adamw_s0_30m` (68.5).
Status of every idea below: **TO-TEST** unless marked otherwise.

## 1. Where the gap lives (measured)

`temp/diag_gap_positions.py` → `logs/diag_gap_positions.log`. Final 100M-token checkpoints, 164k val tokens, EVAL_SEED windows.

| context position | 0 | 1–3 | 4–15 | 16–63 | 64–255 | 256–511 | all |
|---|---|---|---|---|---|---|---|
| lif_exact | 559 | 234 | 178 | 124.5 | 108.7 | 97.7 | 106.4 |
| lif_exact, memory ablated | 559 | 300 | 290 | 299 | 294 | 279 | 287 |
| transformer | 1246 | 211 | 104 | 76.0 | 72.4 | 69.5 | 72.7 |
| **ratio lif_exact / tfmr** | 0.45 | **1.11** | **1.72** | **1.64** | 1.50 | **1.41** | 1.46 |

By target frequency the ratio grows from 1.37 (top-100 tokens) to 1.72 (rarest bucket).
Spikes: 134 active neurons/token in block 0 and 64 in block 1 (of 2048). In block 0, 72% of spikes would not fire from the current token's input alone; they need membrane history.

**Reading.**
- With 1–3 tokens of context the spiking model is only 11% behind, so per-token modeling (embedding, bigram, the spike code) is not the main problem.
- The gap peaks at 4–64 tokens of context. The transformer drops fast (211 → 104 → 76), while the spiking model drops slowly (234 → 178 → 124). The weakness is using *recent* context precisely.
- The long-range gain from the memory is real. From position 16 to 256+, the spiking model improves by 22% and the transformer by only 9%, because at 100M tokens the transformer has no induction heads yet (recall is at chance). With the memory ablated, the curve is flat after position 1–3, so all of the spiking model's context use comes from the GDN-2 memory. The LIF membrane and Engram add almost nothing beyond the bigram.

## 2. Mechanisms (why)

1. **No precise local-context path.** The memory is an induction-style associative store (key = previous token's spikes, read by the current token), so it is not position-aware. The LIF membrane is one fixed exponential filter (β = 0.9) shared by all neurons. The transformer gets position-aware attention at every layer (previous-token heads, n-gram patterns). Linear-attention models close this with **short convs** (every GDN/DeltaNet/Mamba block; GDN reports conv and output gate as "crucial") and **sliding-window or hybrid attention** (SpikingBrain-7B: GLA + SWA 1:1; Based; Samba; NVIDIA Mamba-2-Hybrid).
   - *Our own evidence:* `pre_conv` (09-07) gave the **best val_ppl of any 30M screen** but recall dropped to chance. The conv blurred the key and query pointers (see `neuro_preconv_localattn_null_2026_09_07.md`). That result was leak-era, but t<256 was honest.
   - **Lesson: add local context through a path that does not touch the memory's k/q pointers.**
   - `local_attn` was NULL, but it had **no positional encoding** and reused the memory's prev-token key, so it could not implement positional heads. It was also judged on leaky val_ppl. It has never been tested honestly.
2. **Recipe tuned under the leak.** The transformer's own optimizer switch (NorMuon 0.005 / Adam 6e-4 → AdamW 1e-3) moved it from 87.8 to 68.8, a 1.28× change. Its Muon LR sweep peaked at 0.02, 4× ours. Against the transformer on the *same* optimizer, lif_exact is already only 1.13× behind (99.3 vs 87.8). The spiking LR was set by the leak-era "higher LR hurts recall" finding. Honest lif_exact recall (94% at N64) leaves room to trade.
3. **Thin per-token code (secondary).** 64–134 binary features per token, mostly history-driven. Every published spiking LLM that nears ANN quality trains with **integer / multi-level or ternary spikes** (SpikingBrain, SpikeLM, I-LIF/NASN, SDLLM), then expands them to binary spike trains at inference. We use the hardest variant: strictly binary, one timestep, surrogate gradient.

**Best-of-both estimate.** Suppose the model matched the transformer's local modeling (positions ≤63) and kept its own relative long-range gain from memory (×0.873 at 64–255, ×0.785 at 256+). The position-weighted val_ppl would be about **65 vs the transformer's 72.7, i.e. ≈0.90×.** This is a back-of-envelope number that assumes the effects add; it is not a promise.

## 3. Screen plan (30M, 100M tok, single change vs the lif_exact control)

Recall gate (single-pass N16/32/64): stay ≥ 97/90/80, against the control's 100/100/94. Kill rule: eval2 (32.8M tok) more than 5% worse than the control's 159.7. Report spike rate for every arm.

| # | Change | Type | Recall risk | Why |
|---|---|---|---|---|
| **P0** | Transformer LR check: AdamW 5e-4 and 2e-3 | config (13 min each) | – | Fair baseline. Don't tune only one side. |
| **A1** | NorMuon `muon_lr 0.02`, `lr 1e-3` | config | low–med | Transformer's best Muon LR, 4× ours |
| **A2** | `optimizer: adamw`, `lr 1e-3` | config | low–med | The transformer's exact optimizer |
| **A3** | 2-layer hybrid `[spk, attn]` (`attention_layer_positions: [1]`, `attn_single_residual`, `final_norm`) at the A-winner LR | config | med (spk layer keeps memory) | Direct best-of-both test. About param-matched: attention block ~12.6M vs spiking block ~10.5M |
| **B1** | `ff_conv`: depthwise causal conv (k=4, **identity-init**) on the spikes feeding **only W_ff**. k/q/v/gates untouched | ~10 lines | low (pointers untouched) | Cheapest test of the local-context hypothesis |
| **B2** | SWA read with **RoPE**: window 64, 4 heads, small separate q/k (N→256) + memory's v, zero-init gate, added to r | ~40 lines | low | Position-aware local attention inside every block. The real version of `local_attn`, recall-safe |
| C1 | Integer spikes D=4 (I-LIF) in `lif_exact`: s = clamp(floor(pre/vth), 0, D), subtractive reset, staircase surrogate | kernel | low | The literature's main SNN→ANN gap closer. Still binary spikes at inference (D virtual steps) |
| C2 | PLIF: learnable per-neuron β (init spread 0.5–0.95) | kernel | low | Neurons choose between precise and integrating. Also shows whether membrane history helps at all |
| C3 | Ternary spikes ±1 | kernel | low | 2× information per neuron, stays add-only |
| D1 | Pre-RMSNorm before `to_current` + final RMSNorm | small | low | Standard. Matters more at 9+ layers (v4) than at 2 |
| D2 | GDN output gate: `RMSNorm(r) ⊙ σ(W_g spk)`, low-rank r16 | small | low | GDN and the Gated Attention paper (NeurIPS'25) both show a gain |
| D3 | Separate spiking MLP sub-block (param-reallocated) | medium | low | Our block spends ~3/4 of its params on memory projections. The transformer spends 2/3 on its MLP |

Order: P0 → A1/A2 → A3 + B1 (both are the local-context test) → B2 → C1/C2 → D.
Then stack the winners, run 2 seeds, and do one **600M-token** run against the transformer's 600M (39.6). That run tells whether the gap is a learning-speed lag (the ratio was 1.20 → 1.60 → 1.44 over training) or a capacity ceiling.
GPU budget ≈ 30h of 5080 for singles, less with kill rules. Hold v4 shape decisions until A/B report.

## 4. Don't re-test
- Conv on the key/query path (kills recall).
- Spike rate / sparsity target: rate 0.05 → 0.12 changed eval1 only 185.7 → 183.2.
- FPT K-arms.

## Sources
SpikingBrain 2509.05276 (adaptive threshold, integer spikes, GLA+SWA 1:1) · SpikeLM 2406.03287 (bi-directional elastic spikes) · Adaptive Spiking Neurons 2604.12365 (integer training, spike inference) · Spike-driven LLM 2604.16475 (integer→binary/ternary unfolding) · Gated DeltaNet 2412.06464 (short conv + output gate crucial) · DeltaNet 2406.06484 · Kimi Linear 2510.26692 / Qwen3-Next (3:1 linear:attention hybrids beat full attention) · SymbolicLight 2605.21333 (spike-gated dual path, 7.7% above GPT-2).
