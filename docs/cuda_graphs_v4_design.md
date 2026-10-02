# CUDA graphs for v4 training throughput — design (2026-09-30)

Status: **SCOPED. Recommendation: build Phase 1 (per-spiking-block graphs), but only after the fused-neuron switch is measured and the 1B profile still shows the GPU idle more than ~40% of the time. Phases 2–3 and a fused GDN-2 kernel are deferred behind explicit gates (§5).**
Requested by Glenn 2026-09-30. Companion: `docs/fused_neuron_kernel_design.md` (that switch is a hard prerequisite here).
Measurement scripts (counts only): `temp/census_compiled.py`, `temp/count_neuron_kernels.py`, `temp/kernel_breakdown.py`, `temp/count_graph_breaks.py`.

## 1. What it is

Replay each training micro-step's GPU work from recorded CUDA graphs instead of launching thousands of kernels from Python. v4 is launch-bound: on the v4 draft the GPU was busy only 23–34%, and a step cost ≈770 ms fixed + 54 ms per sample at B8 (5080, 2026-09-27). Under WSL2/WDDM, per-launch cost is higher than on native Linux.

Concretely:
- **Phase 1:** graph each spiking block's forward+backward with `torch.cuda.make_graphed_callables`, the per-layer approach Megatron-Core uses via TransformerEngine for training.
- **Phase 2 (deferred):** capture the whole micro-step, including MQAR, with static MQAR shapes.
- **Phase 3 (deferred):** capture the optimizer step.

**It is NOT:**
- a change to the model math. Phase 1 replays the same kernels in the same order, so it should be bit-identical to the non-graph compiled run;
- a fix for low FLOP efficiency inside kernels. That's the fused GDN-2 kernel (§6, deferred);
- the inductor "cudagraph trees" mode (`compile_mode: max-autotune`). That mode doesn't support gradient accumulation (§3), and v4 uses `grad_accum: 9`.

## 2. Why now vs. why defer

Measured 2026-09-30 (counts, which a shared GPU can't distort), compiled (mode=default), 1B-probe structure at width/4, one main fwd+bwd at B=1:

| | kernels | host syncs | dynamo graphs |
|---|---|---|---|
| FPT neuron (probe as-is) | 7,857 | 80 | 5 |
| `lif_exact_ml` signed L4 | **5,432** | **0** | 5 |
| ↳ of which inductor-fused Triton | 2,869 | | |
| ↳ of which matmuls (mostly small, GDN-2 sub-block loop) | 2,106 | | |
| ↳ eager aten (graph-break glue) / lif / memset | ~400 / 16 / 17 | | |

- At chunk 256, each layer's GDN-2 runs 2 chunks × 16 sub-blocks (`src/delta_chunked.py:355-433`, `sub=16`), so ~32 sequential iterations per layer. That loop generates most of the 5,432.
- MQAR adds a second, separately compiled forward per micro-step (`dynamic=True`, `src/train_gpu.py:538`).
- At GA 9, a 1B step is ≈ 9 × (5.4k main + MQAR) ≈ **60–80k kernel launches**.

**Why not now:** the cheaper lever comes first. The fused-neuron switch alone removes 31% of compiled launches and every host sync, and those syncs would block graph capture anyway. Its effect on GPU-busy % is unmeasured. If the GPU is ≥75% busy after the switch, graphs can buy at most ~1.3× and aren't worth the complexity. **Why soon:** if it is still ≤60% busy, removing the CPU launch path is the largest remaining throughput lever. The 30-day v4 budget (1B / 16B tokens) is throughput-bound.

## 3. Where it plugs into the codebase (and what blocks the obvious route)

**The obvious route, inductor cudagraph trees, is blocked:**
- The memory `neuro_cudagraphs_incompat_2026_08_09` records the old crash: `max-autotune` crashed at 363M with "accessing tensor output of CUDAGraphs that has been overwritten" at RMSNorm (`src/model.py:96`). Two fixes failed, and MQAR bucketing didn't help (2026-09-12).
- Root cause, confirmed by 2026 practice: **cudagraph trees allocate gradient outputs inside the graph pool, so accumulation across replays is unsupported** (pytorch#169545, closed; its documented workarounds, clone + `cudagraph_mark_step_begin`, still fail in the report). rf-detr PR #1479 (2026) disables cudagraphs whenever `grad_accum_steps > 1`.
- The two compiled callables (main + `mqar_model`) sharing one pool make it worse.
- **Conclusion: don't use `compile_mode: max-autotune` for v4.** Use manual capture, where gradients accumulate in eager `AccumulateGrad` or in pre-allocated `.grad`.

**Phase 1 attach points:**
- `src/model.py:1574-1578`: the training single-pass loop calls `block.forward_sequence(x_seq, mems[l], Ms[l], prev_spks[l], ...)` → `(out_seq, mem_f, M_f, spk_seq, final_spk)`. Swap in the graphed callable here **only when** `self.training and torch.is_grad_enabled() and x_seq.shape == graph_shape and not hasattr(block, "qkv")`. MQAR (different T), eval, inference and the KV-cache path keep using the un-graphed block.
- Initial states: zero `mems/Ms/prev_spks` built at `:1454/:1483` each call. The adapter creates them inside the graph (static zeros), so the graph's only input is `x_seq`.
- `src/train_gpu.py:496-506`: `compile_regional` (per-block `torch.compile`) already exists. Phase 1 needs it, because whole-model compile inlines the blocks so they can't be graphed separately. Graph the *compiled* blocks.
- `src/train_gpu.py:493-539`: build the graphs after model/optimizer build and checkpoint resume, before the first step: a warm-up of ≥3 iterations, then `make_graphed_callables` over the tuple of all spiking blocks so they **share one memory pool** (it captures forwards in order and backwards in reverse).
- Neuron: `src/lif_exact.py:206` `@torch.compiler.disable` is fine for manual capture (its Triton launches are recorded). **FPT is not capturable** (`torch.equal` host sync, `src/lif_parallel.py:173`). The `triton_op` wrapper (fused-neuron doc §6) is optional here; it matters only for Phase 2's single compiled graph.

**Constraints the code must respect, each one an audit item:**
1. Nothing Python-side inside `forward_sequence` may vary per step during training. Telemetry is already off in training (`telemetry_eval_only: true`). The `_last_*` attribute writes would freeze at capture time, so audit them (MoE `_last_aux_loss` is None at `n_experts: 1`).
2. Any per-step scalar used *inside* a block must be a device tensor, or it gets baked in at capture. Blocks currently take none; the λ/MQAR ramps live in the loss, outside the graphs.
3. Autocast: graphed callables need `autocast(cache_enabled=False)` (PyTorch CUDA-graphs notes).
4. `act_ckpt` must stay off with graphed blocks; it's eager-only anyway.
5. The `spk_seq` and `out_seq` outputs are static buffers, overwritten on the next replay. They're consumed within the same micro-step (residual, sparsity `sr`), so nothing may keep them across micro-steps. Logging must use detached reductions.
6. Checkpoint resume: re-capture after load. Parameters keep their addresses as long as `load_state_dict` copies in place (it does).

## 4. Cost & risk

| Phase | Dev | GPU | Expected gain | Main risk |
|---|---|---|---|---|
| prereq: neuron switch (fused doc A) | 0 | 1 probe | +10–40% (est.) | — |
| **1: per-block graphs** | ~2–3 days | 30M parity + 1B probe A/B | +20–80% on top, depending on post-switch GPU-busy (est.) | stale/overwritten static buffers; memory pool + regional compile raising peak VRAM (5090 WDDM cliff ~29–30 GB) |
| 2: whole micro-step + static MQAR | ~1 week | + a 30M MQAR-padding recall screen | removes MQAR + glue launches | MQAR distribution change hurts recall; baked-in scalars |
| 3: optimizer capture | ~2 days | probe | ≤ optimizer share (~4% at GA4, less at GA9) | Muon lazy state, LR must become a tensor |
| alt: fused 2048-dim single-head GDN-2 kernel | 2–4+ weeks | screens | fewer kernels and higher kernel efficiency | numeric bugs; big effort |

Failure modes, specific and how to catch them:
- **Stale input.** A static input not refreshed means the same batch repeats and loss drops suspiciously fast. Catch: the first 200 steps must match the non-graph run on the same data.
- **Overwritten outputs.** A static output read after the next replay gives silent garbage. Catch: an in-process gradient comparison over a full GA cycle vs eager.
- **Baked scalars.** A schedule silently freezes. Catch: a test that changing λ changes the graphed loss.
- **VRAM growth.** Lower max B erases the gain. Catch: report peak VRAM.
- **Benchmark artefacts the project has been burned by:**
  - a shared-GPU benchmark gave a bogus 1.31×;
  - the data-bound loader (fixed by MADV);
  - max-autotune warm-up and MQAR recompiles (~55 steps);
  - 5090 CPU-load noise of ±15%;
  - compiled max-autotune runs aren't run-to-run deterministic, so bit-identity tests use mode=default.

## 5. Recommendation & decision gates

0. **Prereq (no build):** switch v4 to `lif_exact_ml` (fused doc §5 A). On an idle 5090, run the 1B probe with FPT vs `lif_exact_ml` and record tok/s **and GPU-busy %** (torch profiler GPU-active share, or `nvidia-smi dmon -s u` SM%, steps 100–300).
1. **Gate for Phase 1:** after step 0, GPU busy **≤ 60%**: build Phase 1. Between 60 and 75%: build only if the B-sweep (B8/12/16 × amb) can't reach ≥75%. **≥ 75%: STOP** (≤1.3× ceiling; spend the time on the batch lever).
2. **Gate for Phase 2:** after Phase 1, the profile shows MQAR + un-graphed glue at ≥25% of remaining step time, **and** a 30M screen shows padded/static MQAR within seed spread on recall N16/32/64 and haystack.
3. **Gate for Phase 3:** optimizer ≥ 5% of step time after Phase 1–2.
4. **Fused GDN-2 kernel:** only if, after graphs, the step is GPU-bound but MFU is still < ~35% with the GDN-2 sub-block matmuls dominating GPU *time* (not just count).

## 6. Ready-to-implement specifics

**Phase 1** (flag `cuda_graph_blocks: true`, default off, so the existing path is byte-identical; requires `compile_regional: true`, `lif_exact: true`, `act_ckpt: false`):
```python
class _BlockGraphAdapter(nn.Module):          # one per spiking block; graph input = x_seq only
    def __init__(self, block, B, T, fpt_K):
        super().__init__(); self.block, self.fpt_K = block, fpt_K
        self.mem0, self.M0, self.spk0 = block.zero_state(B)   # NEW small helper: static zeros built exactly as model.py:1454/1483
    def forward(self, x_seq):
        out, _, _, spk, _ = self.block.forward_sequence(x_seq, self.mem0, self.M0, self.spk0, fpt_K=self.fpt_K)
        return out, spk

# setup (train_gpu, after resume, before step 0), B = batch_size, T = block_size:
adapters = tuple(_BlockGraphAdapter(b, B, T, K) for b in spiking_blocks)
samples  = tuple((torch.randn(B, T, d, device="cuda", dtype=torch.bfloat16, requires_grad=True),) for _ in adapters)
with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
    graphed = torch.cuda.make_graphed_callables(adapters, samples, num_warmup_iters=3)   # shared pool, fwd in order / bwd reversed
raw_model.set_graphed_blocks(graphed, shape=(B, T))   # used at model.py:1575 only under the §3 condition
```
- Keep `opts.zero_grad(set_to_none=True)` as now. Parameter grads come out of the graphed backward as static buffers, and eager `AccumulateGrad` copies or adds them into `.grad`. **If the first micro-step's `AccumulateGrad` can steal the static buffer, use `set_to_none=False`.** The pre-registered gradient test below decides which.
- MQAR, eval and inference automatically use the un-graphed (regional-compiled) blocks.
- **Phase 2 static MQAR** (only at gate 2). Pad each MQAR example to a fixed length by **prefixing real text** (haystack-style). One length (65) for plain MQAR; buckets {128, 256, 512} for the gap variant, one captured graph per bucket sharing a pool. Then capture the whole micro-step (main fwd + MQAR fwd + loss + backward into pre-allocated `.grad`) per the PyTorch "whole-network capture" pattern, minus the optimizer. The NaN guard (`train_gpu.py:907`) stays outside, once per GA cycle. Megatron also needs NaN checks disabled inside its full-iteration graph.

## 7. Pre-registered success criteria

**Correctness (Phase 1), all required:**
- In one process, with the same data, at 30M, compile mode=default, GA 4: parameter grads after one full GA cycle, graphed vs un-graphed, are **bit-identical** (same kernels, same order). Tolerance ≤1e-6 relative only if the shared pool changes the reduction order.
- The 30M loss trajectory over 200 steps is identical to the un-graphed regional-compiled run.
- Changing the sparsity λ mid-run changes the loss.
- Peak VRAM is reported.

**Throughput (Phase 1), 1B probe, idle 5090, interleaved A/B, 2 repeats, median tok/s over steps 100–300:**
- **Adopt if ≥ +20% vs the best non-graph configuration at the same or larger feasible B** (the honest baseline is the batch-lever best, not B8). Peak VRAM must stay ≥ 2 GB below the WDDM cliff.
- **< +5% = noise → revert** (the speed-plan kill rule).
- Report GPU-busy % before and after; it should rise. If tok/s rises but busy % doesn't, suspect a measurement artefact.

**Phase 2 MQAR padding (quality):** at 30M, 2 seeds, recall N16/32/64 and haystack N8/N32 within the control's seed spread; val_ppl within ±1%.

## References (ranked)

**Primary port target (Phase 1):**
1. `torch.cuda.make_graphed_callables` and the PyTorch CUDA-graphs notes: partial-network capture, constraints (no CPU sync, static shapes/addresses, no data-dependent control flow, graph-safe RNG), shared pools, autocast `cache_enabled=False`. https://docs.pytorch.org/docs/2.14/notes/cuda.html
2. Megatron-Core per-layer training CUDA graphs (`--cuda-graph-impl transformer_engine` / `local`, `--cuda-graph-modules`; TE `make_graphed_callables`): https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/cuda_graph.md ; https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/cuda_graph.html

**Phase 2:**

3. Megatron `FullCudaGraphWrapper` / `StaticBufferLoader` (full-iteration capture, excluding optimizer; NaN checks disabled inside): https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.full_cuda_graph.html

**Why not cudagraph trees / max-autotune:**

4. pytorch#169545 (compile + CUDAGraph + gradient accumulation fails): https://github.com/pytorch/pytorch/issues/169545
5. rf-detr PR #1479 (2026; falls back when grad_accum > 1; grads live in the graph pool): https://github.com/roboflow/rf-detr/pull/1479
6. CUDAGraph Trees docs: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html. Inductor graph partition (`torch._inductor.config.graph_partition=True`; splits off non-capturable ops): https://github.com/pytorch/pytorch/issues/125864. Relevant to Phase 2 only.

**Kernel alternative (deferred):**

7. Gated DeltaNet-2 (arXiv:2605.22791, NVlabs, May 2026): the **same erase/write-decoupled recurrence as our GDN-2**, S_t = (I − k_t(b_t⊙k_t)ᵀ) D_t S_{t−1} + k_t(w_t⊙v_t)ᵀ, with fused Triton chunk kernels (C=64, forward-substitution WY solve). Built for H=16, d_k=d_v=128; our single 2048-dim head is unsupported (needs full-d_k reductions for the C×C system plus a tiled state). Code: https://github.com/NVlabs/GatedDeltaNet-2 . Earlier internal note: fla kernels at H≥8 collapsed recall.

**Internal:**

8. `docs/training_speed_plan_2026_09.md` §4.5 (MQAR static shapes; kill rule <5%).
9. Memories `neuro_cudagraphs_incompat_2026_08_09`, `neuro_lif_fpt_nonconvergence_2026_09_27` (v4 profile: launch-bound).
