# Brain principles and the eight videos: what to bring into v4 (2026-10-02, Neuro 48, overnight)

Glenn asked:
- Which ideas from 8 videos would help v4?
- Can we use how human brains work to improve the model?
- Can we make anti-hallucination a selling point?
- Can we build a "System 1" decision engine?
- Can we add an ontology layer?

This builds on `docs/v4_plan_review_2026_10_01.md`. Status: a **proposal**. GPU runs and builds need Glenn's OK, except the evals noted as research.

## TL;DR

1. **Our design already follows the brain's memory layout. Say so in the pitch, and finish it.** The brain runs several memory systems at different speeds; v4 has a matching one for each (§2 table). Two pieces are missing:
   - the hippocampal *episodic index* (HOLA-style cache of surprising moments);
   - *sleep consolidation* (replaying context into the synaptic memory).

   Both are inference/post-training features, built while v4 trains.
2. **Two architecture ideas are worth one more 30M round (r17) before the mid-scale gate.** Both must be in pretraining if we want them:
   - **(a) "One hippocampus, many readers" (YOCO-style).** Upper layers read the memory and attention notes built by the lower layers. This is what DeepSeek V4.1 Flash and Xiaomi MiMo-V3 adopted in September 2026 to cut prefill compute (videos 2 and 7). For us it also targets the recall-at-depth weakness (r14d).
   - **(b) A bidirectional hippocampus.** It also stores "what came *before* X", so the model can answer inverse queries (Fido → dog *and* dog → Fido) from memory, beyond the attention window (videos 7 and 8; the brain's CA3 pattern completion).

   Cost: about one day of build and one night of runs. Launch moves to ~10-06/07.
3. **Anti-hallucination can be a headline feature because we can audit ourselves.** The Merlin-Arthur method (2512.11614) trains a model to answer only when evidence supports it and to refuse otherwise. It cut wrong answers under insufficient context by 18-20 points (1B-32B models). In our model, the "Merlin" evidence comes for free from the audit trail (the ≤k attention sources plus the memory write behind each recall). Product claim: **"certified answers"**. Every answer comes with its evidence, and uncertified answers are flagged.
4. **The System 1 decision engine ("Neuro Reflex") is a strong product line on the same base model.** Jev and Laya read a record and return typed, calibrated decisions in one pass, with no text, so they cannot hallucinate formats. Ours would add:
   - streaming with constant memory (it can watch an endless log);
   - an audit trail per decision;
   - neuromorphic energy;
   - an escalate gate that hands hard cases to System 2 (our generative mode).

   This is post-pretraining work.
5. **Ontology: in the product layer, not in pretraining.** Typed tool schemas plus an OWL/RDFS rule check before any tool call executes ("logic outside, reasoning inside"). Inside the model, the bidirectional memory and the memory audit already give a knowledge-graph view: each recall shows the (subject, relation, object) write it came from.
6. **Video 1's brain constraints map onto concrete choices:**
   - discrete synapses → stage-2 4-bit, then ternary weights;
   - bursts → graded spikes (we have them);
   - refractory period → a cheap kernel flag (r17, low priority);
   - confidence-by-latency → confidence from spike and memory statistics (feeds anti-hallucination).
   - We keep backprop: nobody has matched its quality without it. The brain's local learning lives in our *memory*, which learns at inference time with no backprop.

## 1. Video by video

| # | Idea | Verdict | Where it goes |
|---|---|---|---|
| 1 | Brain constraints: slow spikes, refractoriness, rate/timing/population/burst codes, discrete synapses, slow plasticity, confidence-by-latency, ~20 W | Mostly already ours (spikes, bursts as graded/multi-level spikes, populations, k-WTA sparsity, local Hebbian memory). New: discrete weights (stage 2), refractory flag (test), confidence from internal activity (anti-hallucination) | Stage 2; r17 (low priority); post-training |
| 2 | YOCO prefill split (DeepSeek V4.1 Flash: 40 layers, only 20 run at prefill; MiMo-V3 HySparse2: −80% prefill, −78% KV cache) | **Adopt in spirit**: lower half builds the memory + window KV once; upper half only reads it. Prefill ~2× cheaper; also the "multi-reader hippocampus" for recall at depth | **r17 test**, decide at G3 |
| 3 | Context Language Models (2609.37725): the model edits its own context; RL with a success-gated efficiency reward; suffix cache reuse | Our memory's erase/write gates already are learned context management at the synapse level. Add a **memory-edit API** (erase or pin a fact by key) as an audit and safety feature. RL memory management for agents comes later. Suffix-cache reuse is moot: our state is constant-size | v4.x product; v5 RL |
| 4 | Merlin-Arthur protocol (2512.11614): Arthur answers or rejects; Merlin gives evidence; Morgana injects misleading context; mutual-information (EIF) bound | **Adopt in post-training** with our audit trail as Merlin. Anti-hallucination becomes a measured selling point | Post-training stage + eval |
| 5 | System 1 decision engines (Jev, Laya): typed choice/score/boolean outputs in one pass, calibrated (proper scoring rules), act-vs-escalate gate, ~33 ms | **New product line "Neuro Reflex"** on the v4 base (and a 150M edge version). Brain map: option scorer with k-WTA = basal ganglia action selection; escalate gate = conflict monitoring → System 2 | Post-pretraining product |
| 6 | Neurosymbolic ontology (RDFS/OWL + Pydantic): logic outside, reasoning inside | Product-layer guardrails for agent tool calls. Plus a knowledge-graph view of memory writes | v4.x product layer |
| 7 | Spiking knowledge graph (Brain Simulator): fixed topology, inheritance (multi-hop), automatic inverse relations, sparse firing | Inverse relations → **bidirectional hippocampus (r17)**. Inheritance → 2-hop memory reads; test with a new eval first. Sparse firing → already ours | r17 + new evals |
| 8 | Sequences as linked structure, subsequence reuse, parallel activation search | Already ours, worth saying: memory keyed on the previous token = linked chain; Engram n-grams = reusable subsequences; one memory read = parallel search over all stored links. Possible upgrade: keys from short subsequences (canon taps already mix local context in) | Pitch framing; eval |

## 2. The brain map (pitch slide, and the design checklist)

| Brain system | What it does | Neuro v4 component | Status |
|---|---|---|---|
| Cortex: sparse pyramidal neurons, dendrites, lateral inhibition | Feature computation, ~1-5% active | **DSN** dendritic spiking neurons (apical gate × basal payload, k-WTA) | r16 (C0, L3, X2) |
| Membrane dynamics | Integration over time | LIF membranes (H0 layer; DSN-LIF variant) | r16 X2 |
| Hippocampus CA3: one-shot, pattern completion | Fast episodic memory | GDN-2 Hebbian delta-rule memory (layer 0) | have; bidirectional = **r17** |
| Hippocampal index of salient moments | Keep surprising episodes exactly | HOLA-style surprise cache (β‖e‖ from GDN-2) | v4.x build |
| Sleep replay / consolidation | Fast → slow memory transfer | Sleep mode: offline replay passes over context into the memory (2605.26099); later into weights | v4.x / v5 |
| Thalamus: routing and gating | Picks what to attend to | Top-k sink-free gated attention | r15c (free), r16 |
| Basal ganglia: action selection | Winner-take-all choice | System 1 decision head (Neuro Reflex) | product |
| Prefrontal metacognition | Knows when it doesn't know | Confidence from spike and memory statistics + Merlin-Arthur certified answers | post-training |
| Semantic and lexical memory | Long-term facts and words | Engram n-gram tables (editable) + slow weights | have (hash fixed r16) |
| Discrete synapses, ~20 W | Energy | Stage 2: spike-coded inputs + 4-bit, then ternary weights; event-driven share metric | stage 2; scorecard |
| Neuromodulation (surprise / dopamine) | Gates plasticity | GDN-2 error-driven, gated writes (delta rule); surprise ranks the episodic cache | have / v4.x |
| Homeostasis, refractoriness | Stable, decorrelated firing | Late sparsity ramp (have); per-neuron homeostasis and refractory flag | r17 optional |

**Pitch line:** *"Neuro has a cortex, a hippocampus and a thalamus. It learns while it reads, remembers in constant memory, shows its work, and knows when it doesn't know."*

## 3. Updated plan (changes to `v4_plan_review_2026_10_01.md` §6)

### 3.1 r17 (new; 30M rig, 4 layers, one night on the 5080; builds need OK; after r16)

| Run | Change vs the r16 winner | Build | Question |
|---|---|---|---|
| r17a | **YOCO/multi-reader**: upper attention layers use the KV of layer 1 (shared, computed once), and get a second memory read from H0's memory with their own query | ~1 day: cross-layer KV + read from saved chunk states, with parity tests | Same ppl? Better haystack at depth? (Prefill ~2× is by construction) |
| r17b | **Bidirectional hippocampus**: a second memory in H0 storing "token → the token before it" (key = previous spikes, value = spikes two back) + reverse-MQAR in the aux (half of the queries) | ~half day | Reverse and 2-hop recall with attention ablated; forward recall and ppl unchanged? |
| r17c | Refractory LIF (no spike on the token right after a spike) | kernel flag ~20 lines | Free? Does it change selectivity or the memory keys? |
| r17d | Per-neuron homeostasis (penalise each neuron's rate deviation, not only the mean) | ~10 lines | Fewer dead or over-active neurons → better scorecard |

Decision rule: adopt a feature if ppl is within 1-2%, recall and haystack hold, and its own target metric improves.

**Measured baseline (10-02 02:00, `scripts/kg_recall_eval.py`, CPU, N8, gap 0, 64 queries; chance 12.5%):**

| Model | Forward (full / attn_off / mem_off) | Reverse | 2-hop |
|---|---|---|---|
| r14b | 100 / 100 / 9.4 | 4.7 / 4.7 / 15.6 | 14.1 / 14.1 / 15.6 |
| r15c (top-k16) | 100 / 100 / 17.2 | 1.6 / 0.0 / 14.1 | 12.5 / 18.8 / 6.2 |

- Forward recall is 100% memory.
- **Reverse recall is BELOW chance even with full attention in-window.** The forward memory confidently returns what came *after* the queried value (the next key), which is a candidate but the wrong one.
- 2-hop is at chance.

So our model has an in-context "reversal curse", and so would any model trained only on forward recall. r17b is motivated by data, not only by the brain analogy. It needs the reverse-MQAR aux; the attn_off column shows whether the *memory* (the part that works beyond the window) learned it.

**The transformer has the same curse.** The 3L transformer trained with the same forward-MQAR aux (`screen_tfmr3_adamw_s0_30m`) scores forward 100, reverse **0.0**, 2-hop 10.9. The curse comes from the training signal, not from our memory.

A fair claim therefore trains BOTH models with the reverse and 2-hop aux. Our distinct advantage would be doing it **from memory, beyond the attention window**, where a transformer has nothing.

### 3.2 New evals (no GPU training; build now as research)

- **Reverse-MQAR:** query a value, predict its key, both with and without attention (`attn_off`, memory-only).
- **2-hop MQAR (inheritance):** A→B and B→C stored, ask A→?C.
- **Beyond-window recall:** streaming, once exact streaming exists.
- **Hallucination under insufficient context (Merlin-Arthur style):** wrong-answer rate when the evidence has been masked; EIF.
- **Calibration (ECE)** for the System 1 head.

All go into the scorecard or recall suite, so every candidate gets the same table.

### 3.3 After launch (built while v4 trains)

- **Exact streaming + audit flight recorder** (planned). This is the base for certified answers and the decision engine.
- **Certified answers:** an evidence extractor (top-k sources + memory writes) and a post-training stage with Merlin/Morgana data (helpful evidence, masked evidence, misleading evidence) teaching answer / refuse / ground. It extends the existing ORPO(+why+abstain) stage.
- **Neuro Reflex (System 1):**
  - decision heads (choice/score/boolean) trained with proper scoring rules (Brier/log loss) + temperature calibration;
  - an act-vs-escalate gate;
  - a demo on log and ticket triage with streaming input;
  - a 150M edge variant.
- **Ontology guardrails:** typed tool schemas (Pydantic) + an OWL/RDFS rule check (rdflib/owlready2) before tool execution; a knowledge-graph view of memory writes in the audit viewer.
- **Sleep mode:** offline replay passes that consolidate the window into the memory before dropping it (long-context, constant memory); then a "nightly" consolidation of the memory into the weights for continual learning (v5).
- **Memory-edit API:** erase or pin a stored association by key (delta-rule erase), logged.

### 3.4 Timeline

r16 results (~10-02 evening) → r17 build (~1 day) and run (one night) → G4 1B probe + G3 mid-scale (5090 + 5080) → launch **~10-06/07**, about one day later than the 10-01 plan. If the r17 builds slip, launch without them. Both are add-ons and the r16 candidate stands alone.

## 3.5 r17 implementation design (ready to build on OK)

**r17a: YOCO / multi-reader hippocampus.**
- *Shared KV (cross-layer attention from a bridge layer).*
  - The bridge attention block (layer 1 at 4L, ~L/2 at 1B) keeps its post-RoPE K and V for the current forward.
  - "Cross" blocks above it have only Q and O projections and attend over the bridge's K/V (same top-k, gate and canon).
  - Upper layers then never need their own states for prompt tokens, so prefill can stop at the bridge, as in DeepSeek V4.1 Flash and MiMo-V3.
- *Multi-reader memory.*
  - H0 keeps references to its memory inputs for the current forward (v, k, gates, initial state).
  - Each cross block adds a query projection W_q2 (d → d_mem) and reads r2_t = M_t q2_t from H0's memory, scaled by a sigmoid gate and a small LayerScale.
  - The screen version calls `gdn2_chunked_dense` a second time with q2. This is exact, with no new kernel, at the cost of a second chunk loop.
  - If the feature is adopted, an optimized read from saved chunk-start states replaces it: two batched GEMMs, no loop. Parity test: the sequential reference with q replaced by q2.
- *Why it may enable 2-hop:* a cross block can query memory with a query built from the first read's result (a→b, then b→c). Test it with a 2-hop aux (below).

**r17b: bidirectional hippocampus.**
- A second memory in H0 stores each adjacent pair in reverse, written one step late so a token never matches its own write:
  - key_s = W_kr(spikes at s−1), value_s = W_vr(spikes at s−2), query_t = W_qr(spikes at t);
  - its own low-rank gates and output gate; output added to the forward read.
- On MQAR pairs (k_i at 2i, v_i at 2i+1), the reverse write at s = 2i+2 stores v_i → k_i, so the query v_q returns k_q.
- Cost: one more chunk loop in H0 only (one layer; ~+3–5% at 1B).

**Training signals** (default off; `train_gpu` MQAR aux):
- `mqar_reverse_frac`: the share of aux batches asking value → key;
- `mqar_hop2_frac`: the share asking a → c over [a b] and [b c] blocks.

Without these signals, neither feature has a reason to be learned.

**Runs** (r16 winner, 4L, 100M tokens):

| Run | Config | Question |
|---|---|---|
| r17a | shared KV + multi-reader | ppl, haystack at depth |
| r17a2 | multi-reader only | separates the KV-sharing ppl cost |
| r17b | bidirectional + reverse aux 0.5 | reverse recall (attn_off), forward unchanged |
| r17e | r17a + r17b + reverse 0.3 + hop2 0.2 | the "knowledge-graph" combo: forward / reverse / 2-hop |
| r17c | refractory | |
| r17d | homeostasis | |

Six runs, about one night. Evals: `kg_recall_eval` (forward/reverse/2-hop × full/attn_off/mem_off), haystack, scorecard.

## 4. What we will NOT do, and why

- **Drop backprop for local learning only.** No local rule matches backprop at LM scale. Our local learning is the memory, which learns at inference time.
- **True spike-timing codes inside a token.** That needs several sub-steps per token (×T compute). Burst size (graded spikes) carries the same information at no extra cost.
- **An ontology inside pretraining.** No evidence it helps LM quality, and it is costly. It belongs in the product layer.
- **A transformer → spiking conversion as the main path** (10-01, parked).

## Sources

- YOCO (Microsoft, 2024).
- DeepSeek V4.1 Flash (2026-09; YOCO: 20 of 40 layers at prefill).
- Xiaomi MiMo-V3 / HySparse2 (−80% prefill, −78% KV at 1M tokens).
- Context Language Models: arXiv 2609.37725.
- Merlin-Arthur hallucination bounds: arXiv 2512.11614.
- Laya (ModernBERT-large + decision head; RL with proper scoring rules; ~33 ms) and Jev (TypeSafe, 2026-09).
- HOLA: 2607.02303.
- Language Models Need Sleep: 2605.26099, 2606.03979.
- What attention recalls and recurrence controls in hybrids: 2609.04434. Exact retrieval rides on attention in Qwen3.5/Falcon-H1; in ours it rides on the memory (attn_off unchanged), which is a distinct property.
- How Linear Attention Remembers: 2609.33093.
