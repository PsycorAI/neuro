# n84: back to the vision: v4 audit, v4.5 conversion, v5 plan (2026-10-27)

Glenn 10-27: v4 has drifted from the psycor.ai vision (spiking neurons, a Hebbian synapse that rewrites itself,
brain files that save / ship / compose working memory, low energy, auditable). How do we get back with v4, and how
should v5 be planned? Status: PROPOSAL. Every GPU run needs Glenn's OK.

## 1. Audit: v4 vs the psycor.ai promises (sft_v4_chat cfg, temp/n84_cfg_dump.txt)

v4 = 1.11B params: 12 attention blocks (71.3M each = 856M, 77%; of each, ~50M is the relu2 FFN, ~21M the attention
mixer), ONE memory block (block 6, 84.6M = 7.6%), embed + head 134M, Engram tables 35M.

| psycor.ai promise | v4 today | Verdict |
|---|---|---|
| Neurons quiet until they fire; sparse binary spikes | Every linear layer's INPUT is spike-coded, but as graded single-step counts (8 levels signed, 16 hidden) = closer to 4-bit activations than binary spikes; 77% of inputs exact zero; FFN units fire 5-9%. Membrane (LIF) dynamics only in block 6. | partial |
| Memory = an explicit synapse rewriting itself in real time | One gated delta-rule (Hebbian-type) memory, block 6. Until n83 it did not carry plain-language facts at all (n82). n83 pilot: held-out facts 4/5 to 4k tokens, 3/5 at 8k. | partial, improving |
| Save working memory to a brain file; ship, version, compose | Demo "brains" = LoRA weight adapters; "saved sessions" = text transcripts replayed. The engine CAN save/restore the memory state, but it is not offered as a brain. | drifted (fixable now) |
| "Zeroing the memory nearly doubles perplexity" (350M model) | True of the old 363M model. Not measured on v4; 12L 150M work showed the memory adds ~nothing to ppl. | website claim does not describe v4 |
| Dynamic sparse connectivity (SET) | Not in v4. | dropped |
| Auditable | top-16 attention sources + memory writes can be logged. | partial |
| Low energy | Flat ~540 mJ/word to 16k+ on the 5080 (v4f 1.5-3.1x more). The win comes from bounded attention + constant-size memory and sparse FFN gather, not from spikes as such (spikes only pay on event-driven chips). | real, but not from spiking |

Bottom line: Glenn's summary ("a transformer with one spiking layer") is close. 12 of 13 layers are transformer
blocks with spike-coded inputs and sparse FFNs; the only Hebbian synapse is 7.6% of the weights.

## 2. What the field says (10-27 research)

- Hybrids with MOSTLY delta-rule layers are now mainstream: Qwen3-Next / Qwen3.5 / Kimi Linear use 3 Gated-DeltaNet
  (= our memory's rule) per 1 full-attention layer; Jet-Nemotron-2B keeps only 2 full-attention layers.
- Converting a trained transformer is cheap: RADLADS converts Qwen2.5 7B-72B with 350-700M tokens (<0.005% of
  pretraining); KL-guided layer selection (ICLR 2026) picks which attention layers to keep; Qualcomm 2607.07706:
  delta-style updates approximate softmax attention best; keep a sliding-window path + sinks. Jet-Nemotron
  (PostNAS) keeps the MLPs frozen and replaces attention.
- One plastic layer is too few: TTT-E2E reports that updating more layers at test time is what makes long context
  work; updating only 1 or 3 layers does not.
- Multi-timescale plasticity: Titans (surprise-gated memory), Nested Learning / HOPE (memories updating at different
  rates = fast synapses + slow consolidation), LaCT (large-chunk updates).
- Brain files as states: Cartridges (Stanford) train a compact per-corpus cache offline and compose them at
  inference = the psycor.ai "brain mod" idea, done for KV caches.
- Spiking LLMs: SpikingBrain (7B / 76B) = linear + SWA attention + spiking neurons, converted from Qwen with <2% data;
  spikes simulated on GPUs. Loihi 2 (MatMul-free 370M, preliminary): ~2x less energy, 3x throughput vs an edge GPU.

## 3. Recommendation (in order; each step gated)

A. This week, little or no training
  1. Measure v4's real reliance on its memory: val ppl + chat with memory off (~10 min GPU). Fix the website claim
     if it does not hold for v4 (Glenn's site).
  2. STATE BRAIN FILES on the n83 model: save the memory state after a session, reload it in a fresh session with
     an empty attention window, ask the facts; compose two states. (No training; the n83 eval already does this
     inside one stream.) Brains become memory states again; LoRA adapters are renamed "skills".
  3. Longer n83 run (3x LR, more tokens) + a 50-fact held-out test with exact matching.

B. v4.5 = "convert v4 to a Hebbian majority" (the main step)
  - Replace ~9 of the 12 attention mixers with spiking gated-delta memory mixers (block-6 type), keep ~3 attention
    layers with a 1,024 sliding window, keep ALL FFNs, embeddings, Engram (the $430 bought these; ~75% of params).
  - Init new q/k/v from the attention weights; stage 1 = per-layer output alignment to the attention it replaces;
    stage 2 = KD from v4 itself on chat + text; stage 3 = n83 long-memory + cross-session data on all memory layers.
  - Screen FIRST at 150M on the 5080 (r26o checkpoint exists): ~1 day build + 1 night. It answers our own risk from
    10-01 (r15d: two spiking memories did not stack) before any 1B spend.
  - 1B cost (rough): ~300-500M tokens; ~1-2 days on the 5090, or ~6-10 h on a cloud H200 (~$25-60).
  - Gates: ppl within ~5% of v4; chat OK; beyond-window recall >= n83; memory-off ppl rises sharply (the memory
    is load-bearing again); energy per word still flat.
  - Redo afterwards: chat polish, brains/skills, int8 + prune. Fairness: v4.5 gets extra training tokens vs v4f ->
    give v4f the same extra tokens or disclose.

C. v5 (plan, not a from-scratch pretrain yet)
  - Start from v4.5, not from scratch, unless v4.5 shows the layout is the limit.
  - Layout: ~3 spiking delta-rule memory layers per 1 local-attention layer; attention only local; all long-range
    memory = Hebbian synapses.
  - Three speeds of learning: fast synapses (per token), session memory (larger per-chunk updates, LaCT/Titans style),
    slow "sleep" consolidation of a session's memory into a small slow store / weight delta.
  - Brain files = all plastic states (+ the consolidated slow store); trained from day one with multi-session episodes
    (tell -> save -> reload -> ask) and composition.
  - Spikes: graded spikes everywhere, LIF membranes in memory layers; report event-driven share; binary spikes as an
    ablation only. Ternary weights (BitNet) for deployment = later.
  - Vision scorecard as launch gates: share of plastic layers, memory-off ppl delta, cross-session recall, brain
    composition, energy/word, event-driven share, audit trail per answer.
  - SET rewiring: stays parked (no quality or GPU-energy evidence at 1B).

## Sources
RADLADS 2505.03005; KL-guided layer selection (ICLR 2026) 2512.20569; Analysis-driven linearization 2607.07706;
Jet-Nemotron 2508.15884; Qwen3-Next / Kimi Linear (Raschka, "Beyond Standard LLMs"); TTT-E2E; LaCT 2505.23884;
Nested Learning / HOPE (Google Research blog); Cartridges 2506.06266; SpikingBrain 2509.05276; Loihi 2 LLM 2503.18002;
Hebbian fast weights in transformers 2510.21908.
