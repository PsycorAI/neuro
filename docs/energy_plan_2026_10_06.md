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

## Honest energy with weight reads (10-06 23:20, Neuro 53, `temp/energy_eval_v2.py`)
The results below count arithmetic only. Real chips also read each weight (Horowitz 45 nm, 16-bit: on-chip SRAM
~2.5 pJ, DRAM ~320 pJ per read); dense reads every weight per token, spiking only those of NONZERO inputs.
| model | compute only | + SRAM weight reads | + DRAM weight reads | nonzero inputs |
|---|---|---|---|---|
| r25o (dense src) | 1.39× | 1.38× | 1.38× | 73% |
| r25t (L8 conv) | 5.00× | 3.11× | 1.92× | 52% |
| r25w (L8+q/v+KD, 1.5× thr) | 7.49× | 3.95× | 2.26× | 45% |
- The honest lever is the **nonzero fraction** (one read per nonzero input, whatever its spike count), not the count.
- Demo meter: show the on-chip (SRAM) number as the headline, with arithmetic-only and DRAM as the range.
- Testing: p26k / p26l (7L, thresholds 2× / 3× + q/v + self-KD) on the 5080 overnight.

## Results so far (10-06 15:20)
- r25s (L4, 40M conv) 5.56x, chunk1 +7.8%; **r25t (L8, 80M conv) 5.0x, chunk1 41.67 = dense r25o 41.74, memory intact.**
- 6L: L8 > L4; 2x thresholds 9.7x but +35% at 20M tok; ternary +89% (dropped).
- Self-KD bug (Liger fused CE overwrites the logits in place) fixed: KD now computed before the CE. Reruns p25u / r25w.
- Spike-driven attention (code q, v): +1.4% zero-shot, 5.0x -> 6.2x. Added to r25w (L8 + q,v + KD + 1.5x thresholds, 80M).
- Low-rank LM head (SVD): r512 +6.6%, r384 +18% zero-shot for <=1/3 head saving -> parked.
- Real-text recall beyond the window: r25o only 30.0 vs 28.2 mem-off (xcopy eval). xcopy augmentation: 7L no ppl cost but
  barely helps (7L can't copy in-window either); 12L r25v pending.
- **r25w** (L8 + q/v + self-KD + 1.5x thresholds, 80M): chunk1 44.42 (+6.4% vs dense r25o; ~+14% vs r21b), **7.5x** energy, memory intact (cross-chunk 100/90-100). Trade-off vs r25t (5.0x, +0%). r25y (factorised head) running.
- **r25y head 256**: chunk1 50.27 (+13% over r25w), 8.9x vs original dense (70.2 uJ) -> too costly; head 512 running. Way 2 r25z (14L, memory at 4 + 9) queued after.
- **r25y head 512**: chunk1 48.52 (+9% over r25w), 8.25x vs original dense -> factorised head PARKED at 150M (both ranks: poor ppl/energy trade); revisit at 1B only if needed. Best: r25t (5.0x, +0%) or r25w (7.5x, +6.4%).
- **r25z way 2** (14L, memory at depths 4 + 9, from scratch): chunk1 48.45 (+16% vs r25o), cross-chunk haystack at CHANCE, eval-4 instability spike -> two memory blocks closed-neg as configured; one mid-depth block stays the recipe.
