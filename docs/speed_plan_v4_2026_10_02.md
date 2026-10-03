# v4 speed plan (2026-10-02)

Glenn's ask #3: keep c0's advantages (recall, small circuits) but get close to the transformer's training speed.

## 1. What we measured (5090 with no other job on it, 30M screen shape, B32, eager)

Source: `logs/speedsplit_5090.log` (run 13:42–14:10) and `logs/r17_5090.log`.

| Variant | ms/step | tok/s | vs c0 | Exact math? |
|---|---|---|---|---|
| t4 (fair 4L transformer, AdamW) | 145 | 112.6k | 3.6× | n/a |
| **c0** | 530 | 30.9k | 1.00× | — |
| c0 + batched NorMuon (`muon_batched_ns`) | 538 | 30.5k | 0.99× | yes |
| c0 + `telemetry_eval_only` | 528 | 31.0k | 1.00× | yes (logged numbers unchanged) |
| c0 + AdamW (diagnostic only) | 470 | 34.9k | 1.13× | no — AdamW kills recall |
| c0 + flash attention instead of top-k | 501 | 32.7k | 1.05× | no |
| c0 + `gdn2_chunk_size: 256`, `gdn2_ckpt_chunks: false` (+ bns + tel) | 400 | 41.0k | **1.32×** | yes |
| c0 at B64 | 625 | 52.4k | 1.68× | yes (needs VRAM) |
| c0 compiled | 20,711 | 791 | 0.03× | recompile storm |
| c0 compiled + `telemetry_eval_only` | 5,054 | 3.2k | 0.10× | recompile storm (smaller) |

**Breakdown of the 385 ms gap (c0 vs t4):**
- NorMuon: ~60 ms (~15%). It is needed for recall, so we keep it.
- DSN MLP and top-k attention: ~5% (l3 ≈ c0; flash attention is only +5%).
- **The H0 spiking/GDN-2 layer: the rest (~80%).** It is host- and launch-bound: in the op profile the CPU spends 7.9 s versus 1.2 s of GPU work, with about 4× the kernel launches of t4. Most of those launches come from the GDN-2 sub-block loop.

## 2. Why compile is broken

530 recompiles in 7 steps (`logs/prof_s_c0_compile_diag.log`). Every MQAR aux sequence has a different length (T = 15, 17, 33, 43, 51 …). Two things recompile for each new length:
- the function that resumes after the telemetry graph break, at `src/model.py:853`. Its tensors are guarded with static shapes;
- `hybrid_attn.forward_sequence`, through the RoPE slice guard.

`telemetry_eval_only` removes the first and gives a 4× improvement, but compiled c0 is still 10× slower than eager.

## 3. Plan

| # | Step | Exact? | Cost | Gate / expected |
|---|---|---|---|---|
| A | **Adopt chunk 256 + `gdn2_ckpt_chunks: false` + `telemetry_eval_only: true`** in all new screens (from r18) and in v4 | yes | config only | +32% at 30M. Check VRAM at 1B (no chunk checkpointing uses more memory). |
| B | **Fix compile**: pad MQAR aux sequences to a few fixed lengths (`mqar_buckets`) or `mark_dynamic` the sequence dim, and remove the remaining graph breaks; re-diagnose with `TORCH_LOGS=recompiles,graph_breaks` | `mark_dynamic` yes; buckets change MQAR padding (needs a parity check) | ~½ day of code + a 10-min test | Target: compiled c0 faster than eager (t4 gains 1.7× from compile). |
| C | **G4 1B layout probe** with A+B: tok/s, GPU-busy %, VRAM/max batch, compile, FP8 on/off, against a transformer of the same shape | — | ~3–4 h on the 5090 | This gives the real gap. At 1B, H0 is ~1/13 of the compute and the GEMMs grow, so the launch overhead should matter much less than at 30M. |
| D | **CUDA graphs, Phase 1** (H0 block forward+backward, `docs/cuda_graphs_v4_design.md`) | yes | ~2–3 days | Only if G4 shows the GPU busy less than ~60% of the time. |
| E | **Batch**: B64 gives +68% at 30M (launch-bound). At 1B this is limited by VRAM, so it is decided in G4 together with A. | yes | — | — |

Not worth pursuing: batched NorMuon and telemetry-off alone (no gain at 30M; batched NS may still help at 1B, so G4 checks it); replacing top-k attention with flash attention (+5%, changes the model).

## 4. Status (10-02 afternoon)
- **Glenn approved A, B and C.**
- **A is standard** from r18 and in v4 (recipe table, `docs/v4_plan_review_2026_10_01.md`).
- **B is built.** Two parts:
  - `mqar_pad_to` (train_gpu; unset = byte-identical);
  - the DSN k-WTA moved into a compiler-disabled `_kwta` helper (hybrid_attn), so the per-step k-ramp no longer recompiles.
- **B's parity test passes** (`temp/parity_mqar_pad.py`, CPU, real c0 checkpoint): `_kwta` is bitwise equal to the old code; padding changes logits and gradients by ≤1e-5 (float rounding only).
- **B's GPU compile timing is pending**, run with the r17 queue frozen.
- **CUDA graphs bar:** the gain is at most 1 ÷ (fraction of time the GPU is busy), e.g. 1.33× at 75% busy. Set the bar from the G4 numbers, not a fixed 60%.

## 6. Results (10-02 evening)
- **The compile fix works** (30M, B32): compiled c0 runs at 60.2k tok/s, against 39.1k eager and 30.9k for the original c0. That is 1.95×, with 3 recompiles.
- **G4 at 1B, B8** (the maximum batch for both models):

  | | tok/s at GA1 | tok/s at GA8 |
  |---|---|---|
  | c0-S1 (1.109B) | 5,960 | 10,145 |
  | transformer-S1 (1.096B) | 15,251 | 13,483 |
  | **gap** | 2.56× | **1.33×** |

  At GA1, NorMuon takes ~460 of c0's ~690 ms per step; GA amortizes it. Both GA8 runs reserved 30.8–31.4 GB, which is at the WDDM spill point. A re-check with expandable_segments and at B6 is queued.
- **GPU busy at 1B:** ~79% for c0, ~75% for the transformer, so the CUDA-graph ceiling is ~1.25×.
- **The B2/B4 compile crash is an Inductor SplitScan codegen bug.** It hits the GDN-2 `cumsum(LogA)` (`src/delta_chunked.py:381`) when the outer size is small; B6 and B8 compile fine. It is a compile-time crash only. The exact workaround, a compiler-disabled cumsum, is not applied because v4 uses B≥6.
- **CUDA-graphs prototype** (`temp/proto_graphs.py`): per-block adapters plus `make_graphed_callables`. At 30M, B32, fwd+bwd, with timings noisy:

  | Mode | tok/s | Gradients |
  |---|---|---|
  | regional compile | 131.8k | — |
  | **regional compile + graphs** | **161.3k (1.22×)** | bit-identical |

  - Eager + graphs differs by 6e-3 relative, while an eager rerun gives 0.0. The cause is capture-time kernel choice; this mode won't be used.
  - The 1B timing is queued.
  - Open item: the k-ramp is baked into the graph. Fix with k as a device tensor, or recapture per k bucket.

## 7. Decision for Glenn (asked 10-02)
1. Approve **A** for r18+ and v4. The queued r17h–j runs stay as they are, to keep them comparable.
2. Approve building **B** (compile fix) now, CPU/code only. It needs a 10-minute 5090 timing test, done with the queue frozen as today.
3. Approve **C** (G4 probe) on the 5090 once the r17 queue ends (~17:00).
