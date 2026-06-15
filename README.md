# neuro — a genuinely spiking, Hebbian language model

An honest attempt to build the architecture the *Baby Dragon Hatchling* (BDH) paper
**describes** — not the transformer variant its reference code actually is.

Most "brain-inspired" LMs rename transformer parts. This one implements the mechanisms
for real, and **tests that they are load-bearing**:

- **Spiking neurons** — Leaky integrate-and-fire units (snntorch) emit sparse binary
  spikes, trained with surrogate gradients.
- **Hebbian synaptic memory** — a decaying fast-weight matrix `M` written by
  outer products at inference time: `M_t = λ·M_{t-1} + η·(v_t ⊗ k_{t-1})`, read by
  `r_t = M_t q_t`. This is working memory that lives on the synapses and rewrites in
  real time. Cost is `O(d_mem²)` per token, **independent of context length**.
- **(Phase 2)** SET dynamic sparse connectivity → emergent heavy-tailed / modular graph.

## What this is, and is not

We claim only what we can show. We do **not** claim it rivals GPT-2, that it saves energy
on a GPU, or that biology buys alignment for free. The inference-energy numbers are the
energy an event-driven (neuromorphic) chip *would* use under the standard 45nm model; a
dense GPU will not realize them. What the architecture genuinely offers is **traceability**
(sparse, inspectable synapses) — an alignment *affordance*, not alignment itself.

## Phase 1 results (tiny, runs on CPU)

| Gate | What it shows | Result |
|---|---|---|
| **G3 memory is real** | recall accuracy with vs without the synaptic memory | **0.96 with `M` → 0.05 (chance) ablated** |
| G1 language | beats a bigram baseline on real Llama-3 tokens | see `train_text.py` |
| G2 sparsity | most neuron-timesteps are silent | ~82% sparse (≥ SpikingBrain's 69%) |
| G6 energy | per-token inference energy vs a matched transformer | see `train_text.py` |
| G5 consumer HW | trains & runs on **CPU** | ✓ (~86k-param core) |

The headline: disabling the Hebbian matrix drops induction recall to *exactly* chance —
the memory is doing the work, demonstrably. That is the thing BDH claimed and never showed.

## Run

```bash
python src/train.py         # induction task: G3 (memory necessary) + G2 (sparsity) + G5 (CPU)
python src/train_text.py    # G1 (beats bigram) + G6 (energy vs transformer)
python scripts/viz_synapse.py   # G4: assets/synapse_strengthening.png
python tests/test_phase1.py     # gates as tests (also works under: pytest -q)
python src/train_set.py         # Phase 2: SET sparse connectivity (degree + modularity)
```

## Roadmap

- **P1** spiking + Hebbian core ✓
- **P2** SET sparse neuron→neuron synapses ✓ — degree Gini 0.15→0.22, modularity 0.28 > random 0.26, recall intact (`assets/degree_distribution.png`)
- **P3** scale to 350M, persistent synaptic state across sessions, concept→synapse trace tool
- **P4** full energy accounting + alignment-affordance writeup

## References

SpikeGPT (arXiv:2302.13939) · SpikingBrain (arXiv:2509.05276) · SET (Nat. Commun. 2018,
s41467-018-04316-3) · Schlag et al. 2021, *Linear Transformers Are Secretly Fast Weight
Programmers* · Miconi et al. 2018, *Differentiable Plasticity*.
