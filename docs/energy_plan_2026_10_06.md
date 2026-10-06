# Energy plan — spike the whole model, keep the memory (2026-10-06, Neuro 51)

Glenn (10-06): raise the energy savings, keep a reasonable ppl and excellent memory; pursue "three ways to bring the
spiking back" starting with (1) spiking MLPs + (3) post-pretrain conversion, then (2) more memory layers.

## Where the energy goes (measured, `temp/energy_eval.py`, 45 nm MAC 4.6 pJ / AC 0.9 pJ, neuromorphic estimate)
r25o (13L: 12 attention + memory at depth 6): **1.39× less than the same model run dense** (r21b also 1.39×).
Per token: MLP input projection 130 µJ, LM head 116, QKV 98, attention gate 33, attention output 33, memory input 11.
All of these read DENSE inputs, so they are multiply-accumulates. The sparse 5% MLP hidden is already cheap.

## Lever: spike-code every linear input (`spike_code`, `src/spike_code.py`)
Signed multi-level integer spikes with a learned per-channel threshold (SpikingBrain integer spike counts / SpikingMamba
TI-LIF): an input becomes a count s in {-L..L}; a linear layer then costs |s| accumulates per input and nothing for
zeros. Sites: qkv (+ gate), attention output, MLP input, MLP hidden (unsigned), LM head, memory input.
Zero-shot (p25o, no training): every site costs +1–4% ppl at ~0.9–1 spike/input; the MLP hidden needed calibration on
its nonzero values (fixed: +4%, 0.055 spikes/input). Expected after training: ~5× on those layers at ~1 spike/input,
more as the spike count drops.

## Steps (status)
1. **Conversion (ways 1 + 3)**: late phase from a trained ckpt with all sites coded — r25s (12L, +40M tok, 5090) and
   p25s (6L, +20M tok, 5080). Judge ppl vs dense, cross-chunk memory, energy. RUNNING.
2. **Fewer spikes**: raise thresholds (calibration scale) and/or a spike-count penalty; try L 2 / ternary (L 1).
3. **Spike-driven attention**: code q and k so QK^T becomes accumulates (attention core ~11 µJ at T 512).
4. **Spiking from early training** (way 1 alone) vs conversion, at 6L.
5. **Way 2**: a second memory layer (e.g. depths 4 and 9) once 1–4 settle; check memory, ppl, energy.
6. **v4**: r25o layout + chunked attention + a final spike-conversion phase (last ~10%).

## Decision rules
ppl within ~5% of the dense model; cross-chunk recall ≥ 90%; energy ≥ 5× vs dense. Report energy honestly as a
neuromorphic-hardware estimate (not GPU watts).
