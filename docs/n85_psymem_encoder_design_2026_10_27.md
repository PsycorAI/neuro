# Psymem spiking sparse encoder — design (2026-10-27, Neuro 85)

**Status: PROPOSED, awaiting Glenn's go.** Follows Glenn's decision (10-27): the whole memory product stays synaptic /
neuromorphic, with no vector-search hybrid.

## Why

- Psymem's semantic cue today = bge-small (a dense transformer) → random FlyHash → k-winners-take-all.
  - FlyHash is a lossy compression of the bge vector, so at best it ties exact vector search.
  - Measured on LongMemEval-S, semantic channel alone, r@10: cosine 84.2; FlyHash 256/16k 77.0; 4,096/65k signed + rate-coded
    81.3 (`logs/n85_fly_sweep.log`).
- bge-small is the last non-synaptic part of the product.
- A trained encoder that emits sparse spike codes **directly** fixes both problems. Trained on conversational-memory
  retrieval, it can also **beat** the generic embedder. Today's baseline uses off-the-shelf bge (disclose this).

## What it is

- Text → small spiking encoder → expansion layer of N neurons (e.g. 65,536) → k-winners-take-all (k ≈ 1–4k).
  - Out comes a sparse code; it is wired into Psymem's 1-bit / graded synapses exactly like today's FlyHash code.
  - Recall stays accumulate-only.
- **Size:** 5–30M parameters (2–30 MB in the Psyfile, versioned).
  - Target: CPU encodes ≥ 1k turns/s on a laptop.
  - Encoder upgrade = replay the diary into new synapses.
- **Built from our own spiking blocks** (neuro codebase, tiny scale): graded single-step spikes, and LIF where it helps.

## Training (no LoCoMo / LongMemEval data: those stay test sets)

0. **Quick step: a Hebbian-learned expansion layer (BioHash-style, Ryali et al. 2020).**
   - Replace random FlyHash weights with weights learned by a local Hebbian / competitive rule on unlabelled turns, still
     on top of bge.
   - Minutes of compute; tests how much learned codes recover before we build the full encoder.
1. **Distil** bge-small's similarity structure: code overlap ↔ cosine on ~2–5M sentences/turns (fineweb-edu, ultrachat_200k
   turns; both in the HF cache).
   - Gets us to "ties vector search, fully synaptic".
2. **Contrastive memory retrieval:** InfoNCE on (question, evidence turn) pairs with in-batch and hard negatives.
   - Pairs are synthetic: Llama-3.2-3B writes the questions that ultrachat / Capybara turns answer, including temporal
     and multi-hop ones.
   - This is where we can beat the generic embedder.
3. **Sparse-code tricks:** straight-through k-WTA, or a relaxed top-k with an anneal; a FLOPS-style regularizer
   (SPLADE, credit) for even neuron use; no dead neurons.

## Compute

- **5080:** synthetic question generation, ~200k turns × 2 questions. Llama-3B batched, a few hours.
- **5090:**
  - distillation, hours;
  - contrastive training, hours to a day per run;
  - plus 2–3 size variants.

## Checks (in order)

1. Semantic channel alone, r@10 on LME and LoCoMo ≥ bge cosine (84.2 LME).
2. Full Psymem ≥ hybrid_cc on both benchmarks with one shared config and several seeds, using a held-out split for any
   tuning.
3. End-to-end QA (`psymem/qa.py`) ≥ the ordinary-search systems.
4. Speed and size on a laptop CPU; synaptic events per recall.

## Risks

- Synthetic questions may not match real memory questions. Mitigation: mix the question types; check on both benchmarks.
- Spiking encoders train less stably than dense ones. Mitigation: start graded / single-step, as in v4; add LIF later.
- A fine-tuned dense embedder would also improve. The honest claim = "a synaptic memory that matches or beats
  off-the-shelf vector search", plus a disclosure line.

## Prior art to credit

FlyHash (Dasgupta et al. 2017); learned FlyHash / BioHash (Ryali et al. 2020); SPLADE (learned sparse retrieval);
spiking sentence encoders (e.g. SpikeBERT / SpikingBERT lines); HippoRAG; HeLa-Mem.
