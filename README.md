# PsycorNeuro: a genuinely spiking, Hebbian language model

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#status)
[![Runs on CPU](https://img.shields.io/badge/runs%20on-CPU-success.svg)](#)

A language model built on three biologically-grounded mechanisms - with tests
that prove each one is load-bearing, not just decorative:

- **Spiking neurons** - Leaky integrate-and-fire units (snntorch) emit sparse
  binary spikes, trained with surrogate gradients.
- **Hebbian synaptic memory** - a decaying fast-weight matrix `M` written by
  outer products at inference time:
  `M_t = λ·M_{t-1} + η·(v_t ⊗ k_{t-1})`, read by `r_t = M_t q_t`.
  Working memory that lives on the synapses and rewrites in real time.
  Cost is `O(d_mem²)` per token, **independent of context length**.
- **SET dynamic sparse connectivity** - neuron→neuron synapses are evolved
  during training (prune weakest, regrow random), producing an emergent
  heavy-tailed and modular topology rather than a dense layer.

Most "brain-inspired" LMs just rename transformer parts. This one implements
the mechanisms for real, and **ablates each one to prove it's doing work**.

## Status

**Active development  early stage.** Small-model results below are verified
and reproducible. Larger-scale runs (~9M and ~20M params on real text) are in
progress. Quality competitive with a similarly-sized transformer has **not yet
been demonstrated**; that is the next milestone, not a current claim.

## What this is, and is not

We claim only what we can show.

- ✅ The synaptic memory is **inspectable** and **load-bearing** - ablating it
  collapses in-context recall to chance.
- ✅ The inference cost is **constant in context length**; transformer attention
  grows.
- ✅ The model is **small enough to train and run on CPU** at proof-of-concept
  scale.

We do **not** claim:

- That it rivals a state-of-the-art transformer on language quality.
- That it saves energy on a GPU. The inference-energy numbers below are the
  energy an event-driven (neuromorphic) chip *would* use under the standard
  45 nm operation-count model. A dense GPU will not realize them - and may run
  the model *slower* than a transformer of equal size.
- That "brain-inspired" means "aligned." What the architecture genuinely offers
  is **traceability** - sparse, inspectable synapses and a saveable
  synaptic-state file you can audit. That is an alignment *affordance*, not
  alignment itself.

## Phase 1 results (tiny, runs on CPU)

| Gate | What it shows | Result |
|---|---|---|
| G1 language | beats a bigram baseline on real Llama-3 tokens | spiking ppl 179 < bigram 255 |
| G2 sparsity | most neuron-timesteps are silent | ~82% sparse (≥ SpikingBrain's 69%) |
| **G3 memory is real** | recall accuracy with vs without the synaptic memory | **0.96 with `M` → 0.05 (chance) ablated** |
| G4 viz | working memory forms in-context | P(correct) 0.06 → 0.82 at the repeat |
| G5 consumer HW | trains & runs on **CPU** | ✓ (~86k-param core) |
| G6 energy | per-token inference energy vs a matched transformer | 1.6–5.1× cheaper, flat in context length |

**The headline:** disabling the Hebbian matrix drops in-context recall to
*exactly* chance. The memory is provably the mechanism doing the work - this
is not a transformer in disguise.

## Phase 2 results (SET dynamic sparse connectivity)

At 2 % initial connection density, evolved over ~100 SET rounds:

- degree **Gini 0.15 → 0.22** (heavier-tailed than the random start)
- **modularity Q = 0.28** > random Erdős–Rényi baseline 0.26
- induction recall **stays at 0.99** - topology evolution does not break the task

See `assets/degree_distribution.png`.

## Run

```bash
python src/train.py             # induction task: G3 (memory necessary) + G2 (sparsity) + G5 (CPU)
python src/train_text.py        # G1 (beats bigram) + G6 (energy vs transformer)
python scripts/viz_synapse.py   # G4: assets/synapse_strengthening.png
python tests/test_phase1.py     # gates as tests (also works under: pytest -q)
python src/train_set.py         # Phase 2: SET sparse connectivity (degree + modularity)
```

## Roadmap

- **P1** spiking + Hebbian core ✓
- **P2** SET sparse neuron→neuron synapses ✓
- **P2.5** GPU proof-of-concept on consumer hardware (in progress) - ~9M and
  ~20M-parameter runs on real text, head-to-head against a matched transformer
- **P3** scale up, **persistent synaptic state across sessions** (saveable
  "brain" files), concept→synapse trace tool for auditability
- **P4** full energy accounting and an alignment-affordance writeup

## References

· SpikeGPT (arXiv:2302.13939) 
· SpikingBrain (arXiv:2509.05276)
· SET (Nat. Commun. 2018, s41467-018-04316-3)
· Schlag et al. 2021, *Linear Transformers Are Secretly Fast Weight Programmers*
· Miconi et al. 2018, *Differentiable Plasticity*

## License

The source code and model weights in this repository are licensed under the **Apache License 2.0**. Use of these materials is subject to the terms and conditions outlined in the [LICENSE](./LICENSE) file. 

## Intellectual Property & Licensing

**Psycor.ai™**, **PsycorNEURO™**, **PsycorSAGE™**, and **PsycorEdge™** are trademarks of Psycor.ai, established May 2026. All rights reserved.

## Contact

For commercial inquiries or specialized licensing regarding the SGNE architecture, please contact the PsycorAI team at psycoredge@gmail.com.
