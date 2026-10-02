# Code review: interpretability, auditability, neuromorphic fit (2026-10-01)

Scope:
- `src/model.py` (SpikingHebbianBlock, SpikingHebbianLM), `src/hybrid_attn.py`, `src/engram_v2.py`, `src/lif_exact.py`, `src/energy.py`, `src/train_gpu.py` (build, optimizer, loss)
- configs `screen_r14b_*` (best 30M), `flagship_v3_363m` (live), `flagship_v4_363m.draft`

Companion plan: `docs/interpretable_spiking_plan_2026_10_01.md`.

Legend: **I** = interpretability/audit, **N** = neuromorphic fit, **Q** = quality, **S** = speed. Status: BUG / TO-TEST / TO-BUILD / QUEUED (r15) / NOTE.

## Findings, most important first

### 1. BUG: Engram v2 hash aliases token ids (I, Q). Fix in v4; screen first.

`_hash_order_head` keeps the **low** bits of a linear hash: `(Σ prime_i · id_i) & (K-1)`. The low 13 bits of that sum depend only on the low 13 bits of each id. So with K = 8192, ids `x` and `x ± 8192` land in the **same bucket in every table**; the multi-head hashing does not help.

Measured with `temp/check_engram_hash.py`: flipping the top id bit leaves the bucket unchanged at 100% of positions, in all 8 tables.

Affected:
- every 30M screen (vocab 16384: Engram sees id mod 8192, a 2-way merge);
- the **live v3 flagship** and the **v4 draft** (vocab 32768, K = 8192: a 4-way merge of every token).

K = 65536 with vocab 32768 is clean.

Why it matters:
- Engram's value is understated.
- An "auditable lookup table" whose entries systematically mix unrelated tokens cannot be audited.

Fix:
- Multiplicative hashing done right: take the **high** bits, `h = ((Σ p_i·id_i) mod 2^32) >> (32 − log2 K)`.
- Put it behind a flag such as `engram_v2_hash: hi`. Fresh runs only, because it changes the function.
- For v4, also use K ≥ 65536 (8 tables × 65536 × 64 = 33.5M params, almost no compute).

Test: r14b with the hash fixed vs r14b vs r15f (no Engram).

### 2. The spiking block's output LayerNorm (I, Q, N). QUEUED r15d; a follow-up flag is TO-BUILD.

`out = LayerNorm(r + ff)` is added to the residual at unit scale (model.py, end of forward_sequence). This causes three problems:
- **Q.** A spiking layer cannot start near-identity, which is the likely reason a 2nd spiking layer hurts (H1). `depth_scaled_init` scales `W_ff` *inside* the norm, so the norm undoes it, and **r13b was a no-op**.
- **I.** The norm couples all active neurons: a spike's effect on the residual depends on every other spike at that token, so attribution is non-linear. With no output norm, each spike writes a fixed vector, column `W_ff[:, i]` (a "dictionary atom", exactly like an SAE decoder). That makes per-neuron attribution exact and linear.
- **N.** One less normalisation (division and sqrt) per layer.

Plan:
- r15d tests LayerScale with the norm kept (`spk_layerscale`, built).
- Next, a flag `spk_out_norm: false` (no norm, LayerScale only) to get linear atoms.

### 3. Exact streaming (token-by-token) inference is impossible today (N, I, S). TO-BUILD before v4 deployment work.

Three things reset on every call:
- the LIF membrane (`lif_exact` kernels start at `mem = 0`, and `forward_sequence` ignores the carried `mem`);
- canon taps (zero-padded per call);
- Engram n-grams (zero-padded per call).

So the model is not token-decomposable (known for LIF since 09-30). This hurts three ways:
- An event-driven chip processes a stream, so a model that needs window re-forwards is not deployable as a neuromorphic stream.
- A per-token audit trail should come from the same computation that is deployed.
- Window re-forwarding wastes inference compute.

Fix:
- Give the LIF kernels an optional `mem0` input.
- Keep the last k−1 inputs for canon and the last n−1 tokens for Engram in the state dict.
- Add a parity test: streaming token by token must equal the window forward.

Training stays window-based.

### 4. Attention layers are dense transformer layers (I, N). QUEUED r15a/b/c/g.

Each layer is dense softmax over every causal key plus a dense 4096-unit GELU MLP, and the MLP is two thirds of its params. No audit hooks existed.
- Built today: `attn_topk`, `attn_ffn: false`, `SpikeGLU` (`attn_ffn_spikeglu_k`), and `_attn_weights()`, which returns per-head attention weights for audits.
- For v4, also add a **gated-attention output gate**: a sigmoid gate on the attention output (Qwen, NeurIPS 2025 best paper). It improves stability and scaling and **removes attention sinks**. Sinks would otherwise waste a top-k slot in every query and distort audits. Our memory already has the equivalent (`mem_out_gate`).

### 5. Spiking neurons are not more selective than dense MLP units at n = 2048 (I). TO-TEST (S3).

`temp/diag_neuron_selectivity.py`, at matched activity: top-1 token share 0.093 (spiking) vs 0.128 (transformer GELU); p90 0.235 vs 0.434. Interpretability needs to be designed in:
- **width** (BDH uses n ≈ 32k with d = 256; ours is n = 2d);
- **lateral inhibition** (k-WTA);
- a **measured scorecard**.

### 6. Only about 3% of operations are event-driven (N). TO-BUILD (metric), design TO-TEST.

Using `energy.py`'s own model for r14b, per token: about 49M dense MAC vs about 1.4M spike-driven AC. The MACs come from:
- the attention layers (qkv, o and the MLP: 25M);
- the LM head (16.8M);
- the spiking layer's dense input projection (2.1M);
- the memory state ops (3.1M).

`energy.py` also counts the GDN-2 gates as full rank (they are rank 16 now) and ignores Engram.

Fix:
- Update `energy.py`.
- Report an **event-driven share** in the scorecard.

Designs that raise the share:
- SpikeGLU (its down-projection is event-driven);
- spike-coded Q/K/V;
- spike-encoded layer inputs (the "membrane shortcut" of spike-driven transformers, with integer-trained, binary-at-inference spikes, cf. arXiv 2604.16475);
- neuron-keyed memory (sparse reads and writes).

### 7. No memory audit trail (I). TO-BUILD (small).

The GDN-2 erase, write and decay gates (`b_seq`, `w_seq`, `alpha_seq`) are computed every step but only summarised as telemetry.

Fix: an eval-only `audit` mode (off by default) that keeps per-token write strength, erase and decay. Add an approximate **read attribution**: for each read, the past writes ranked by key·query match × retention. That is a "memory attention map", and it gives the scorecard metric "share of correct recalls traceable to one logged write". `temp/diag_haystack_pathways.py` already computes retention, so it is a starting point.

### 8. Multi-level spikes are effectively ternary (N, S). TO-TEST (config only).

r14b fires 9.5% of the time, and 9.4% of that is level ±1. Levels 2–4 are almost unused, but the kernel's surrogate still evaluates 8 bumps per element. A cheap test: `lif_levels: 1` with `lif_signed: true`. If quality matches, adopt ternary spikes, giving one event per neuron per token: cleaner neuromorphic semantics and a simpler backward. This is the integer-train / binary-infer question in the literature.

### 9. Dead parameters and config clutter (I, code auditability). NOTE; cleanup needs OK.

Under GDN-2:
- `W_beta` and `W_alpha` (from `delta_rule` / `decay_gate`) are built but never used.
- `lam` and `fpt_K` are ignored (`lif_exact`).
- `use_fpt` now just selects the sequence path, so the name is a misnomer.

About 15 closed-negative features are still live branches in `model.py`: titans, SDM, OSDN, DeltaProduct, log-linear, local_attn, pre_conv, SWA read, ff_conv, spk_mlp, Hebbian-MoE, MoR/ponder, the recurrent router, fracconn, markov, the confidence head, and Engram v1. They make the model hard to audit as code and feed the "build drift" bug class (bitten 7×).

Recommendation: a lean v4-only model module, or archive the dead branches once v4 is frozen. Not while runs are live.

### 10. Recipe drift: the v4 draft is the old recipe (Q). Rebuild in part 3.

`flagship_v4_363m.draft.yaml` has:
- 8 spiking layers + 1 attention layer (attention at position 4);
- `use_fpt` FPT neurons (no `lif_exact`; FPT never converges), sparsity target 0.15;
- no signed / levels / pre_norm / mem_out_gate / canon / ramp;
- Engram K = 8192 (finding 1).

Flagships also differ from the screens:
- RMSNorm vs LayerNorm block outputs;
- `embed_scale`, MTP and z-loss 1e-4 (the z-loss A/B was null).

v4 must be rebuilt from the screened recipe and checked with `/audit-config`.

### 11. Speed (S).

| | Issue | Status |
|---|---|---|
| a | The spiking block makes 7–8 separate GEMMs on the same spike tensor: `W_k` (on the shifted spikes, equal to a shift of `W_k` on the spikes), `W_v`, `W_q`, `W_ff`, `W_b`, `W_w`, `W_fdecay`, `W_og`. Fuse them into one GEMM plus a shift. This is an exact reassociation with fewer launches, and the model is launch-bound (GPU ~30% busy at 1B). | TO-BUILD, parity-provable |
| b | The gap curriculum doubles step time (an extra 16 × ~500-token batch through the GDN chunk loop) and costs 13–15% ppl on signed stacks (r12, r14c). | TO-TEST: late/short dose |
| c | Top-k attention uses explicit T×T fp32 scores with no fused kernel. Fine at 30M with T = 512; v4 needs a chunked or flash-style top-k, or window + top-k. | NOTE |
| d | `lif_exact(_ml)` is `torch.compiler.disable`, so there is a graph break per spiking layer under compile. Fewer spiking (GDN) layers per model means fewer breaks; spiking MLPs without memory need only one kernel. | NOTE |
| e | Per extra spiking+memory layer, wall time ~1.6× at 30M (r13b 80 min vs r13a 49 min; launch-bound GDN loop). Attention layers are nearly free (flash). | NOTE, drives the v4 layout |

### 12. Baseline fairness (Q, claims). QUEUED r15e.

The transformer baselines had no `data_independent_windows` (~3%) and no Engram. For the final external claim, also give the transformer **canon** layers: Physics of LMs 4.1 shows canon layers help transformers too.

### 13. No interpretability or audit metrics in the eval suite (I). TO-BUILD (M1 in the plan).

## What was built today (r15, default off, bit-identical when off, `temp/parity_r15_flags.py`)

- `attn_ffn: false` drops the attention-layer MLP.
- `attn_ffn_spikeglu_k` / `attn_ffn_hidden` give the k-WTA SpikeGLU MLP.
- `attn_topk` gives hard top-k attention, with a KV-cache path that matches the full forward.
- `spk_layerscale` adds a per-channel gain on the spiking block output.
- `HybridAttentionBlock._attn_weights()` is an audit helper.
- Pre-change backups are in `temp/*.bak_r15`.
