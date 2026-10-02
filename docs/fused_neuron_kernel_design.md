# Fused spiking-neuron kernel for v4 — design (2026-09-30)

Status: **SCOPED. Recommendation: no new kernel to build. The fused neuron already exists (`lif_exact_ml`); switch v4 to it by config, and fold the one remaining integration task into the CUDA-graph scope (next).**
Requested by Glenn 2026-09-30 ("focus first on the fused neuron kernel; after that dev work, scope CUDA graphs").
Measurement scripts (counts only, safe on a shared GPU): `temp/count_neuron_kernels.py`, `temp/kernel_breakdown.py`, `temp/count_graph_breaks.py`.

## 1. What it is

The request was to fuse `lif_parallel`'s K=10 fixed-point (FPT) iterations into one Triton kernel. **That kernel already exists, in a better form.** `src/lif_exact.py` computes the exact sequential LIF: one forward kernel walks t = 0..T-1 per neuron block, and one backward kernel does the reverse scan with the atan surrogate. `lif_exact_ml` (multi-level, signed, per-neuron beta) is the neuron in every current 30M screen (`lif_exact: true`, required by `lif_levels`/`lif_signed`; `src/model.py:561`).

Fusing FPT itself is **rejected**. FPT never converges at T=512 (always hits K=10, and 21–31% of its spikes differ from a true LIF). The honest 30M comparison also favours the exact kernel: lif_exact 99.3 vs FPT 136.8 val_ppl (memory: bf16-leak entry, 2026-09-27). A faster FPT would be a faster version of the worse neuron.

What remains is small:
- **(A)** switch the v4 configs from FPT to `lif_exact_ml`. This is config only; `configs/flagship_v4_363m.draft.yaml` and `temp/probe_1b_b8.yaml` still use FPT.
- **(B)** register the kernel as a `torch.library.triton_op` so `torch.compile` stops graph-breaking at it. The payoff is enabling CUDA graphs and activation checkpointing under compile, so it belongs to the CUDA-graph scope.
- **(C)** occupancy tuning, only if a profile shows the kernel matters.

**NOT in scope:** the GDN-2 memory recurrence. That is where the launches actually are (see §3), and it is the CUDA-graph scope's main target.

## 2. Why now vs. why defer

Measured 2026-09-30 on the 5080 (kernel and sync COUNTS, eager, B=1, T=512; valid on a shared GPU because they don't depend on load):

| | kernels | host syncs (DtoH) |
|---|---|---|
| one neuron call, FPT K=10 (N=2048 or 9728) | 277 | 10 |
| one neuron call, `lif_exact_ml` signed L4 | **12** | **0** |
| v4 draft 310M (9L, 8 spiking) fwd+bwd, FPT | 21,564 | 80 |
| v4 draft + `lif_exact_ml` fwd+bwd | 19,477 | **0** |
| 30M signed 2L (current stack) fwd+bwd | 3,843 | 14 (none from the neuron) |

Forward-only breakdown, v4 draft + `lif_exact_ml` (4,843 kernels over 8 spiking blocks, ≈600 per block):
- the neuron is ≈24 kernels per block (**~4%**);
- the named projections and gate norms are ≈70 per block;
- the remaining **≈500 per block (~85%) is the `gdn2_chunked_dense` Python loop**: 8 chunks × 4 sub-blocks plus `unit_lower_inv`'s log2(C) steps (`src/delta_chunked.py:435`, `:355`, `:39`). That loop issues 560 exp kernels, 480 cuBLAS gemv calls and ~1.7k generic elementwise kernels per forward.

`torch._dynamo.explain` on v4 + `lif_exact_ml`: 5 graphs, 4 breaks, **all caused by `lif_exact_ml`'s `@torch.compiler.disable`**. That's a few µs of Python per step, negligible on its own.

Conclusions:
- **Correction to the 2026-09-30 depth analysis.** The 30M 3L slowdown (1.7× wall time for +32% FLOPs) is not the neuron. Those screens run eager (`compile: false`) with `lif_exact_ml`; the cost is the GDN-2 chunk loop's launches.
- **(A) Adopt now.** Switching v4 cuts the neuron's launches by ~95% and its host syncs to 0, and it's the better neuron on quality. ~~Estimate 0–15%~~ **REVISED same day by the COMPILED census** (`temp/census_compiled.py`, 1B-probe structure at width/4, compile mode=default, B=1 fwd+bwd): FPT **7,857 kernels + 80 syncs → lif_exact_ml 5,432 + 0 (−31% of compiled launches)**. Compile fuses most of everything else, so the eager FPT neuron is ~1/3 of what remains. If v4 is purely launch-bound, the ceiling is ~1/0.69 ≈ +45%. No-sync FPT (0 gain compiled) removed syncs but not kernels, so it isn't evidence against this. **Revised estimate: +10–40% on compiled 1B. Unmeasured.** This is also a hard prerequisite for any CUDA-graph capture (FPT's `torch.equal` syncs can't be captured); see `docs/cuda_graphs_v4_design.md`.
- **(B) and (C): defer.** (B) only pays off through CUDA graphs or activation checkpointing under compile. (C) only if the neuron shows up in a 1B compiled profile.

## 3. Where it plugs into the codebase

- `src/lif_exact.py:120-137` forward kernel `_lif_ml_fwd_kernel`; `:140-168` backward `_lif_ml_bwd_kernel`; `:171-203` `_LIFExactML` autograd.Function. Grid `(B, cdiv(N, 256))`, BLOCK=256, `:180-183`, `:197-198`.
- `src/lif_exact.py:206` `@torch.compiler.disable` on `lif_exact_ml` → the graph break (B replaces this).
- `src/model.py:719-729` neuron selection inside `forward_sequence`. It imports at call time, which keeps monkeypatching and profiling easy. Call sites: `:753`, `:763` (split path), `:1041` (spk_mlp).
- `src/model.py:561` `lif_levels`/`lif_signed`/`lif_learn_beta` require `lif_exact: true`.
- `src/train_gpu.py:183-185` flag plumbing; `:493-509` compile (whole-model `:508`, or regional per-block `:504` with `compile_regional`); `:538` separate `mqar_model` compile (`dynamic=True`, the CUDA-graph blocker).
- **Configs to change for (A):** the v4 config, when it is re-sized to 1B. `temp/probe_1b_b8.yaml` for the measurement.
- **Constraints:**
  - `src/lif_parallel.py` must stay byte-identical, because v3 checkpoints and post-training import it.
  - Membrane starts at 0 on each call. This is the same for both kernels, so no inference/KV-cache behaviour changes.
  - `lif_exact_ml` saves fp32 `pre` for backward: at 1B B8 that's 8·512·9728·4 B ≈ 159 MB per layer, ≈1.3 GB over 8 layers. That's comparable to FPT's saved tensors, so it doesn't change the batch-size lever.

## 4. Cost & risk

| Item | Dev cost | Compute | Main risk | Mitigation |
|---|---|---|---|---|
| (A) v4 config switch | ~0 | one probe A/B on the 5090 (~1 h) | quality at 1B is unscreened (evidence is 30M only); confounding with the other v4 changes | single-flag A/B on the probe; quality evidence = the 2-seed 30M screens |
| (B) triton_op wrapper | ~0.5–1 day | 30M parity runs | silent autograd/compile mis-composition (PyTorch docs warn about autograd.Function + compile) | `register_autograd`, bit-identity tests in eager and compiled |
| (C) kernel tuning | ~1 day | isolated bench | wasted effort if the neuron isn't hot | gate on the 1B profile (≥5% of GPU time) |

Failure modes seen before in this project:
- A benchmark on a shared GPU produced a bogus 1.31×.
- 30M screens were 2× data-bound before the MADV_RANDOM fix.
- Compiled max-autotune runs aren't run-to-run deterministic.
- MQAR recompiles until ~step 55.
- 5090 CPU load adds ±15% noise.

## 5. Recommendation & decision gate

1. **Build nothing new for the neuron.**
2. **(A) Adopt `lif_exact: true, lif_levels: 4, lif_signed: true` in the v4 config** at the 1B re-size. Measure the throughput effect once the 5090 is free (fold into handoff item 3, the 1B B8 probe): two runs of `temp/probe_1b_b8.yaml`, FPT vs `lif_exact_ml`, same B, idle GPU, interleaved A/B, 2 repeats each.
3. **(B) goes into the CUDA-graph scope** (Glenn's next item). The spec is in §6 so that scope needn't re-derive it.
4. **(C) only if** the 1B compiled profile shows `_lif_ml_*` kernels ≥5% of GPU time.
5. **The real launch lever is the GDN-2 chunk loop (~85% of per-block kernels).** Candidate attacks: CUDA graphs (no math change), or a fused single-head large-state GDN-2 Triton kernel. fla's kernels don't apply: head dim ≤256 would force H≥8, which collapsed recall. **Scope this next under CUDA graphs.**

## 6. Ready-to-implement specifics

**(A) config:**
```yaml
lif_exact: true
lif_levels: 4
lif_signed: true
```
The v4 1B shape and sparsity lambda come from the 30M screens (lambda Pareto: to-do "SIGNED SPARSITY PARETO").

**(B) triton_op wrapper** (flag `lif_triton_op: true`, default off, so the existing path is byte-identical). This is the pattern from the PyTorch tutorial:
```python
from torch.library import triton_op, wrap_triton

@triton_op("neuro::lif_ml_fwd", mutates_args={})
def lif_ml_fwd(cur: Tensor, beta: Tensor, vth: float, gamma: float, levels: int, signed: bool) -> tuple[Tensor, Tensor, Tensor]:
    spk = torch.empty_like(cur); pre = torch.empty(cur.shape, device=cur.device, dtype=torch.float32)
    memf = torch.empty(cur.shape[0], cur.shape[2], device=cur.device, dtype=torch.float32)
    wrap_triton(_lif_ml_fwd_kernel)[(B, cdiv(N, 256))](cur, beta, spk, pre, memf, T, N, vth, gamma,
                                                       LEVELS=levels, SIGNED=signed, BLOCK=256)
    return spk, memf.to(cur.dtype), pre            # pre returned so setup_context can save it

@triton_op("neuro::lif_ml_bwd", mutates_args={})
def lif_ml_bwd(gspk, pre, beta, gmemf, vth: float, pa: float, gamma: float, levels: int, signed: bool) -> tuple[Tensor, Tensor]: ...

def setup_context(ctx, inputs, output): ctx.save_for_backward(output[2], inputs[1]); ctx.args = inputs[2:]
def backward(ctx, gspk, gmemf, _gpre): return lif_ml_bwd(...)  # -> (gcur, gbeta.sum(0) or None, None, ...)
lif_ml_fwd.register_autograd(backward, setup_context=setup_context)
```
Keep `@torch.compiler.disable` on the old entry point.

**(C) tuning knobs** (bit-identical: each neuron's arithmetic is unchanged):
- BLOCK ∈ {64, 128, 256} and num_warps ∈ {1, 2, 4}. At 1B B8 the grid is 8×38 = 304 programs on 170 SMs; smaller blocks give more programs and better latency hiding.
- `tl.range(0, T, num_stages=2..4)` to software-pipeline the `cur` loads (they don't depend on the membrane).
- Fuse the sparsity statistic: per-program partial sums of |s| replace a separate reduction over `spk`.

## 7. Pre-registered success criteria

- **(A) throughput:** on an idle 5090, compiled `probe_1b_b8`, median tok/s over steps 100–300 (after the MQAR recompile warm-up), 2 interleaved repeats each. **Adopt-as-throughput-lever if ≥ +10% vs FPT at equal B, and peak VRAM ≤ FPT + 1 GB.** Below +10%, adopt anyway, for quality, but don't count it as a speed lever. **Quality:** no 1B criterion is possible without a long run. The evidence is the 30M honest comparison (99.3 vs 136.8) plus the 2-seed signed screens; the v4 run's early loss curve is a sanity check only (different neuron, so not comparable to v3).
- **(B)**:
  - forward outputs and grads bit-identical to `_LIFExactML` (eager);
  - 30M eager loss identical over 100 steps;
  - `torch._dynamo.explain` shows 0 breaks from the neuron;
  - compiled tok/s not worse.
- **(C)**:
  - `triton.testing.do_bench` with warmup ≥200 ms (the default 25 ms under-reports by ~30%, triton#2306), rep ≥1000 ms, L2 flush, idle GPU, clocks noted;
  - **≥1.5× on (B8, T512, N9728) fwd+bwd with bit-identical outputs**, and a measurable end-to-end gain (≥3%), else revert.

## References (ranked)

**Primary / port target:**
1. `src/lif_exact.py` in this repo. It is the fused kernel (built 2026-09-27; `temp/test_lif_exact.py`, `temp/test_lif_ml.py` parity vs float64 reference).
2. SpikingJelly multi-step neuron CuPy/Triton backends: the same design class (one thread per neuron loops over T; fused forward+backward with surrogate). https://github.com/fangwei123456/spikingjelly ; Fang et al., *Science Advances* 2023, https://www.science.org/doi/10.1126/sciadv.adi1480 (arXiv:2310.16620).
3. Temporal fusion for GPU SNN training: 5–40× vs SNN libraries on A100. arXiv:2408.00280, code https://github.com/EMI-Group/snn-temporal-fusion.

**Integration (B):**

4. PyTorch "Using User-Defined Triton Kernels with torch.compile" (triton_op, wrap_triton, register_autograd; autotune restrictions): https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html. torch 2.11 here has `torch.library.triton_op`.

**Eval rigor:**

5. `triton.testing.do_bench`: https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html ; short-warmup under-reporting: https://github.com/openai/triton/issues/2306
6. Kernel timing practice (CUDA events, L2 flush, clock locking): https://www.jan.ai/post/how-we-benchmark-kernels ; https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/

**Lineage / rejected:**

7. FPT, arXiv:2506.12087. Rejected: non-convergent at T=512, not a LIF, loses on honest val_ppl.
8. Parallel Spiking Neurons (no reset), arXiv:2304.12760. Already reachable via `lif_reset: 0`.
