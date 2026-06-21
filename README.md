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

## Hardware notes

The Phase 1 / Phase 2 scripts (`src/train.py`, `src/train_set.py`, `src/train_text.py`)
run **comfortably on CPU** — no GPU required.

The Phase 2.5 GPU trainer (`src/train_gpu.py`) uses backpropagation through the full
unrolled time loop. Peak VRAM scales with `batch_size × block_size × d_mem²`, so the
default configs are tuned for **16 GB consumer GPUs** (RTX 4080 / 4090 / 5080-class):

| Config preset | `block_size` | `batch_size` | `grad_accum` | Effective batch | Peak VRAM | Throughput (5080) |
|---|---|---|---|---|---|---|
| `phase25_a.yaml` (~9 M params) | 128 | 128 | 1 | 128 | ~6 GB | ~250-500 k tok/s |
| `phase25_b.yaml` (~20 M params, default) | 512 | 16 | 4 | 64 | ~9 GB | ~50-100 k tok/s |
| 24 GB GPUs (3090 / 4090 / 5090) | 512 | 32 | 2 | 64 | ~16 GB | ~9 k tok/s (similar) |
| CPU smoke test (`--smoke`) | 16 | 4 | 1 | 4 | n/a | seconds per step |

### Training speedups (on by default)

The default `phase25_a.yaml` / `phase25_b.yaml` configs bundle every speed +
sparsity improvement we've validated. The math is unchanged — same
architecture, same memory mechanism, same energy story — but training runs
roughly 5-10× faster on a dense GPU than the original sequential implementation.

- **`use_fpt: true`** — Fixed-point Parallel LIF (arXiv:2506.12087). Replaces
  the sequential time-stepped LIF cell with ~K=10 parallel-scan iterations.
- **Chunkwise Hebbian** (auto-enabled inside `use_fpt`). Rewrites the per-token
  `M += v ⊗ k` recurrence as a chunkwise matmul (TFLA-style).
- **`sparsity_lambda: 0.5`, `sparsity_target: 0.05`** — small penalty that
  keeps spike rate near 5% silent (matches the published 96.3% silent baseline).
- **`compile_dynamic: true`** — `torch.compile` with `dynamic=True` so
  shape changes (batch / block) don't trigger a full recompile.
- **`optimizer: muon`** — validated ~10% better than AdamW at this scale.

All are configurable in the YAML if you want to compare. Other useful knobs:

```yaml
optimizer: muon           # Newton-Schulz orthogonalized momentum; 25-35%
                          # faster training-to-equal-loss vs AdamW on 2D
                          # hidden-layer matrices. Fused AdamW handles
                          # everything else (embeddings, biases, LayerNorm,
                          # head) automatically.
muon_lr: 0.02             # Muon-typical learning rate
tie_weights: true         # share embed + head weight matrices.
                          # Requires d == d_mem. Saves ~43% of params on a
                          # 20M model and usually improves perplexity at
                          # small scale.
```

The default optimizer is `adamw` (with `fused=True` on CUDA). Set
`optimizer: muon` to opt into the Muon path.

### Inference speedups

For inference, **`torch.compile(model, mode="reduce-overhead")` is the
single largest practical win** -- CUDA Graphs eliminate per-step launch
overhead which otherwise dominates the spiking model's runtime:

```python
import torch
model = torch.compile(model, mode="reduce-overhead", dynamic=False)
# first call: 30-60s graph capture
# subsequent calls: ~33x faster than eager for the spiking model;
#                   ~6x faster for the transformer
```

See `scripts/bench_inference.py` for a head-to-head benchmark.

### Troubleshooting

- **Throughput drops sharply after hours of training, or `vram reserved` creeps
  toward your card's limit.** Peak working set is spilling into shared GPU memory
  (PCIe-bound, ~20× slower than VRAM). Reduce `batch_size` and increase `grad_accum`
  proportionally — gradient quality is unchanged. The structural fix (gradient
  checkpointing through the time loop) is on the Phase 3 roadmap.
- **First eval at `eval_every` takes 5–30 minutes.** This is a one-time
  `torch.compile` graph capture for the eval kwarg signature; subsequent evals
  reuse the cached graph and complete in seconds.
- **CUDA `device not ready` on resume.** The on-disk inductor cache at
  `/tmp/torchinductor_*` may have artifacts from a different allocator/driver state.
  Clearing it (`rm -rf /tmp/torchinductor_*`) and relaunching forces a fresh compile.

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
