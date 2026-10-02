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

## 5. Decision for Glenn (asked 10-02)
1. Approve **A** for r18+ and v4. The queued r17h–j runs stay as they are, to keep them comparable.
2. Approve building **B** (compile fix) now, CPU/code only. It needs a 10-minute 5090 timing test, done with the queue frozen as today.
3. Approve **C** (G4 probe) on the 5090 once the r17 queue ends (~17:00).
