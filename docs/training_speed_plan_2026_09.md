# Speeding Up flagship_v3 Without Paying Quality — Design Doc

**Date:** 2026-09-10
**Status:** Research + ranked plan. NO code written yet. Probes queue behind the
90M capacity-confirm on the 5090.
**Motivation:** `configs/flagship_v3_363m.yaml` is a **15.6-day** commitment on
the 5090 (1.0e10 tokens). Every 10% of throughput is ~1.5 days. This doc asks
what can be bought without touching model quality, and explicitly separates the
levers that are **quality-neutral by construction** from the ones that need a
screen.

---

## 1. Diagnosis: the run is overhead-bound, NOT FLOP-bound

Arithmetic from the config (not yet a measurement — see §5 step 1):

```
B=18 x block_size 512                    =   9,216 tok / micro-batch
x grad_accum 4                           =  36,864 tok / optimizer step
max_tokens 1.0e10 / 36,864               = 271,267 steps
at ~7,400 tok/s -> 36,864/7400           =    4.98 s / step
271,267 x 4.98 s                         =  ~15.6 days        <- matches config ETA
```

So a micro-batch costs ~1.245 s. Dense fwd+bwd work is roughly
`6 x N_nonembed x tokens`. Total params 363M; the 32768 x 1280 embedding is
~41.9M (and the head another ~41.9M if untied), so call non-embedding compute
params ~280M:

```
6 x 280e6 x 9,216  = 1.55e13 FLOP / micro-batch
1.55e13 / 1.245 s  = ~12.4 TFLOP/s achieved
RTX 5090 bf16 dense ~ 210 TFLOP/s
                   -> ~6% MFU
```

Even **doubling** the FLOP estimate to absorb checkpoint recompute, MTP depth-2,
Engram-v2 and the state ops lands at **~12% MFU**. A well-optimized dense
transformer runs 35-45%.

**Caveat (do not overclaim):** this is config arithmetic, not a profile, and a
chunked-recurrent model legitimately runs below dense-transformer MFU — the
chunk loop is sequential, the kernels are small, and the state traffic is
bandwidth-bound. But 6-12% is low enough that there is real headroom, and it
says something important about *where*: **we are not matmul-bound, so
quantization (FP8/int8) is the wrong place to look.**

### Why it is overhead-bound, structurally

1. **72 sequential chunk-steps per forward.** block_size 512 / `gdn2_chunk_size`
   64 = 8 chunks, x 9 layers. Each is a handful of small kernels — launch-
   overhead territory. Under **WSL2/WDDM** launch overhead is worse than native
   Linux.
2. **Every chunk is re-run in backward.** `src/model.py:698,711,715` pass
   `ckpt_chunks=self.training and self.gdn2_ckpt_chunks`; the default is `True`
   (`src/model.py:218,900`, `src/train_gpu.py:153`) and flagship_v3 does not
   override it. So all 72 chunk-forwards execute a second time every step.
3. **CUDAGraphs are off** (`compile_mode: max-autotune-no-cudagraphs`) — which is
   precisely the mitigation for launch overhead. Root cause in §4.5.

### Already optimal — do not re-litigate

- Newton-Schulz **already runs in bfloat16**, 5 steps (`src/muon.py:20,26`). The
  "NorMuon 28% slower" note is not a per-step throughput tax to reclaim here;
  with `grad_accum: 4` the optimizer is amortized over 4 micro-batches.
- Eval + checkpoint overhead: 135 evals (`eval_every 2000`) + 68 checkpoints
  (`ckpt_every 4000`) ≈ **~1 hour over 15.6 days (~0.3%)**. Leave it alone.

---

## 2. The key distinction: quality-neutral vs needs-a-screen

The whole point of this doc. Three levers change **no numerics at all** and so
need only a throughput/VRAM probe, never a quality screen:

| Lever | Why quality is untouched |
|---|---|
| `gdn2_ckpt_chunks: false` | Activation checkpointing is mathematically the identity — it trades memory for recompute. Same values, modulo nondeterminism. |
| `gdn2_chunk_size` 64 -> 96/128 | Chunked-WY is **exact** (per `neuro_gdn2_dense_chunked`); chunk size is a pure blocking strategy. |
| B / grad_accum rebalance | Hold effective batch at 72 seqs and move the split. No normalization layers depend on batch. |

Everything else (MQAR bucketing, WSD, token budget) **does** move the recipe and
must be screened.

---

## 3. Ranked levers

| # | Lever | Effort | Risk | Est. win | Quality screen? |
|---|---|---|---|---|---|
| 1 | **Profile first** | ~2 h | none | sets the rest | n/a |
| 2 | `gdn2_ckpt_chunks: false` at lower B | hours | low | 10-20% | **no** |
| 3 | `gdn2_chunk_size` 64 -> 96/128 | hours | ~none | 5-15% | **no** |
| 4 | B/accum rebalance at fixed effective batch | hours | none | 5-10% | **no** |
| 5 | MQAR static shapes -> unlock CUDAGraphs | days | med | 10-25% | yes (recall) |
| 6 | Re-test WSD now that `head_wd` exists | ~1 d probe | med | *stoppability* | yes |
| 7 | Right-size `max_tokens` / cosine horizon | decision | quality | up to 30% | judgement |

---

## 4. Detail

### 4.1 Profile before optimizing (do first)

`scripts/` has **no profiling tool**. Worth ~2 h: `torch.profiler` over ~50
steps at the real 363M shape, after compile warmup, broken into main fwd / main
bwd / MQAR pass / optimizer / eval, plus **GPU-idle-gap** measurement to confirm
the launch-bound hypothesis (if the GPU is idle between kernels, §1 is right and
#5 is the big one; if it is busy, #2/#3 dominate). Reorder everything below by
what this shows.

### 4.2 Turn off chunk checkpointing (#2)

Already a config flag, so this is a one-line probe. Removing it deletes a full
recompute of the memory path in backward but costs VRAM, forcing B down.

**The argument this is net-positive:** `neuro_flagship_b18_sizing` measured
B=16 -> B=18 as only **+13.2%** tok/s. That is a flat part of the batch curve —
so batch is cheap to give back, and recompute is expensive to keep.

Grid: `(ckpt=on, B=18)` vs `(ckpt=off, B=10/12/14)`. Watch the **WDDM cliff at
~29-30 GB** (`neuro_flagship_b18_sizing`) and keep >=2 GB headroom; note
post-resume fragmentation can add ~2 GB over a fresh start.

### 4.3 `gdn2_chunk_size` sweep (#3)

32 -> 64 already helped once (`neuro_gdn2_dense_chunked`). Bigger chunks mean
fewer sequential steps and fewer launches, at more work and memory per chunk —
there is an optimum. Sweep {64, 96, 128}. Because chunked-WY is exact this is
the single cheapest possible win. Sanity-check the loss curve matches for the
first few hundred steps (fp noise only).

### 4.4 B / grad_accum rebalance (#4)

Effective batch is 72 sequences. `(B=12, accum=6)`, `(B=9, accum=8)` etc. hold
that constant while repositioning on the throughput curve. Compose with #2 —
the ckpt=off optimum will sit at a different B.

### 4.5 MQAR static shapes -> retry CUDAGraphs (#5)

`neuro_cudagraphs_incompat_2026_08_09` says "don't re-attempt without MQAR
restructure." **This is that restructure**, and the root cause is concrete, not
mysterious:

`mqar_aug_batch` (`src/train_gpu.py:51`) builds `[k1 v1 ... kN vN kq]` — length
**2N+1** — with **N resampled from [4,32] every step** (`src/train_gpu.py:775-790`).
Variable shape forces a *second* compiled wrapper over the same `raw_model` at
`mode="default", dynamic=True` (`src/train_gpu.py:468-471`), i.e. no
max-autotune, up to ~29 graph variants, and no possibility of CUDAGraphs.

MQAR is only ~1.8% of tokens (B=18 x ~37 avg tokens vs 36,864/step) but likely
far more of wall-clock, because dynamic recompiles and short-sequence launches
are proportionally terrible. So fixing it buys the MQAR pass *and* unblocks
CUDAGraphs for the main graph.

Options: (a) bucket N to {4,8,16,32} -> 4 static graphs; (b) pad to 2*32+1=65
with a mask/pad token; (c) hold N fixed across a window of steps.

**Risk:** (a) and (b) perturb the MQAR distribution, and MQAR range [4,32] is a
3-seed CONFIRMED recall win (`neuro_mqar_range_30m_2026_09_03`) — do not
disturb it blind. Gate on a 30M recall screen showing bucketed ≈ uniform[4,32]
on single-pres N16/32/64. That is a ~1.5 h 5080 job.

### 4.6 Re-test WSD now that `head_weight_decay` exists (#6)

The most underrated item. WSD was rejected because flagship_v2 diverged at step
18k (6.6% of budget) via the **head-norm runaway**
(`neuro_flagship_v2_wsd_diverged_2026_08_30`). `head_weight_decay: 1.0` is the
**confirmed mechanistic fix for exactly that mechanism** — 2-seed 30M + a 363M
probe through v2's death zone — and it now also covers `mtp_proj`, which
`neuro_head_weight_decay_2026_09_08` identified as the actual v2 trigger
(+34%, leading the head).

This is not a per-step speedup. It buys **stoppability**: anneal on demand and
get a clean endpoint whenever you choose, instead of a fixed 15.6-day
commitment. Because v2 died at 6.6% of budget, a ~1-day probe settles it.

### 4.7 Right-size the token budget (#7)

The elephant. `max_tokens: 1.0e10` at 363M is **~27.5 tok/param** — deliberately
overtrained — and cosine to 1e10 means the run **cannot be truncated** without a
badly-annealed model (the config says exactly this). A *completed* cosine to
7e9 beats a *truncated* cosine to 1e10. That is ~30% of wall-clock riding on a
horizon choice made once.

Evidence the useful signal may arrive early: v2's salvaged `step_0160000` hit
val_ppl **19.33**, already beating v1's 19.61
(`neuro_head_weight_decay_2026_09_08`). Worth an explicit decision rather than
drift — and #6 turns it into a non-decision.

---

## 5. Recommended sequence

1. **#3 + #4** — free, quality-neutral, fastest to answer. Short real-shape
   throughput probes (~200 steps past compile warmup each).
2. **#1 profile** — with the cheap wins banked, measure the true breakdown.
3. **#2** — ckpt=off grid, positioned by the profile and #4's curve.
4. **Then decide #5 vs #6/#7** on the profile: big GPU-idle gaps -> #5;
   otherwise the budget/schedule question (#6/#7) is worth more than kernel work.

**Hardware constraint:** 363M-shape throughput probes **must run on the 5090** —
the flagship shape needs 27-29 GB and will not fit the 5080's 16 GB. The 5080
can only serve the 30M recall screen for #5.

---

## 6. Success criteria (pre-registered)

- **Throughput probes (#2/#3/#4):** steady-state tok/s at the real 363M shape,
  measured over >=200 steps *after* compile warmup (max-autotune first steps are
  very slow); report tok/s + peak VRAM. Accept only with >=2 GB headroom under
  the ~29-30 GB WDDM cliff. No quality screen required (§2) — but confirm the
  loss curve tracks the control for a few hundred steps as a smoke test.
- **#5 MQAR bucketing:** 30M recall screen, bucketed vs uniform[4,32], single-
  pres N16/32/64 within seed spread; only then retry `max-autotune` with
  cudagraphs and re-measure.
- **#6 WSD:** 363M probe must clear v2's death zone (step ~18k) with head-norm
  and `mtp_proj` norm receding, val_ppl tripwire untripped.
- **Kill rule:** anything under 5% throughput gain at the real shape is noise —
  do not adopt it and do not spend a second round on it.

---

## 7. Explicitly deprioritized

- **FP8 on Blackwell (sm_120).** Only pays when matmul-bound; §1 says we are
  not. d=1280 is also below where FP8 typically wins, and the custom chunk
  kernel would not benefit without separate work. Related caution:
  `reference_blackwell_int8_regression` (different mechanism — bitsandbytes
  dequant — but the same lesson: quantization did not pay when compute-bound).
- **DDP across 5090 + 5080.** Heterogeneous 32/16 GB, no NVLink, WSL2 NCCL, and
  the 5080 is needed for screens. Realistically +40-60%, not 2x, at high
  complexity, and it stalls the flagship every time the 5080 is borrowed. The
  current split (flagship on 5090, screens on 5080) is the better use.
- **Eval / checkpoint overhead.** ~0.3% of the run. Touching it is wasted effort.
- **Reducing MTP depth / Engram-v2 / NorMuon.** These are quality recipe
  components, not overhead; cutting them is a quality trade, not a speedup, and
  belongs to a different decision.

---

## 8. References

- Config under study: `configs/flagship_v3_363m.yaml`
- Code: `src/model.py:218,698,711,715,900`; `src/train_gpu.py:51,153,439-452,468-471,775-797`;
  `src/muon.py:20,26`
- `neuro_flagship_b18_sizing` (B=18 = +13.2% over B=16; WDDM cliff ~29-30 GB)
- `neuro_gdn2_dense_chunked` (chunked-WY exact; 32->64 chunk perf fix)
- `neuro_cudagraphs_incompat_2026_08_09` (cudagraphs + MQAR; two fixes failed)
- `neuro_mqar_range_30m_2026_09_03` (MQAR [4,32] 3-seed confirmed — do not disturb blind)
- `neuro_head_weight_decay_2026_09_08` (head_wd 1.0 confirmed; mtp_proj was v2's trigger)
- `neuro_flagship_v2_wsd_diverged_2026_08_30` (why WSD was rejected)
- `reference_blackwell_int8_regression`, `neuro_normuon_renorm_fix`
