# Pre-flagship code audit — 2026-08-08

Independent read of the pretrain + eval path before the 363M flagship commits.
Companion to `plan_2026_08_05.md` and `polish_recommendations_2026_08_07.md`;
this doc only lists things those two do **not** already cover, plus three
measured numbers that change the plan.

Everything below was measured on the idle 5080 or read from the live
`p4_seed1` log — nothing touched the 5090 run.

Probes written: `temp/audit_routing.py`, `temp/probe_flagship_ratio.py`,
`temp/probe_flagship_vram.py`, `temp/probe_gate_signs.py`.

---

## A. Three measured numbers that change Phase 5

### A1. The flagship is ~12–13 days, not 8.5

Measured step-time ratio 363M/276M (same GPU, same batch, eager, full
loss incl. MTP + sparsity + optimizer step):

```
276M (d1152/n2752/L8)   params=276.57M  step=2072.1 ms
363M (d1280/n3072/L9)   params=363.54M  step=2519.3 ms
ratio = 1.216x   (param ratio 1.314x)
```

1.216× is a *lower* bound — at B=2 the probe is launch-bound, so the
compute-bound ratio sits nearer the 1.32× FLOP ratio. `p4_seed1` is running
at 8,000–8,800 tok/s, so the flagship lands at **~6,200–6,800 tok/s**:

| budget | tok/param | wall clock at 6,500 tok/s |
|---|---|---|
| 7.0B (plan) | 19.3 | **12.5 days** |
| 5.5B | 15.1 | 9.8 days |
| 4.5B | 12.4 | 8.0 days |

The plan's 8.5-day figure implies ~9,500 tok/s, which is faster than the
*276M* model runs today with gate-norm on. Gate-norm is now mandatory (W1),
so that throughput is not recoverable. **This is a budget decision, not a
bug** — 7B at 12.5 days is fine if that's the intent, but it should be an
intent, not a surprise on day 9.

### A2. `block_size: 768` does not fit at batch 16 — answers Q2 and Q4

Peak-allocated slope fit (B=1/2/4, eager; conservative vs compiled):

| shape | T | fixed | per-sample | **extrapolated B=16** |
|---|---|---|---|---|
| 276M | 512 | 2.97 GB | 1.037 GB | 19.6 GB |
| 276M | 768 | 2.92 GB | 1.573 GB | 28.1 GB |
| 363M | 512 | 3.87 GB | 1.291 GB | **24.5 GB** |
| 363M | 768 | 3.81 GB | 1.977 GB | **35.4 GB** |

Calibration: the 276M/512/B16 row predicts 19.6 GB allocated; `p4_seed1`
logs `vram 5.8/21.5 GB` (allocated-at-log / reserved) — consistent. The
model also retrodicts the 2026-08-05 `dxgkio_make_resident: -12` incident
(276M, gate-norm, batch 28 → 2.97 + 1.037×28 = **32.0 GB**, exactly the
wall).

So:
- **363M @ 512, B=16 → ~24.5 GB allocated / ~27 GB reserved.** Fits with
  ~5 GB of headroom. Not generous given this project's WDDM history, but
  workable.
- **363M @ 768, B=16 → 35.4 GB. Will not fit.** The only 768 option is
  B=8 / grad_accum=8 (~19.6 GB), which costs throughput on top of A1's
  12.5 days *and* changes the batch size that W4 validated.

**Recommendation: lock `block_size: 512` for the flagship.** Longer context
is a real lever, but it is not affordable in the same run as 19 tok/param —
bank it for the 1B, where the shape gets re-derived anyway.

### A3. Decay-gate saturation is entirely one-sided, and telemetry can't see it

`_gate_sat_stats` reports `frac(|logit| > 4.595)`, which conflates two
opposite failures. Measured on `p4_seed0/step_0012000.pt` (393M tok):

```
 blk |  sat% | LO(never-forget) | HI(fast-forget) | mean f |  std | mean alpha
   0 |   36% |            35.4% |            0.1% |  -4.10 | 1.34 |     0.9601
   1 |    2% |             1.7% |            0.0% |  -2.57 | 1.09 |     0.9023
   2 |    5% |             4.6% |            0.0% |  -2.89 | 1.16 |     0.9188
   4 |    1% |             1.2% |            0.0% |  -2.24 | 0.95 |     0.8717
   5 |   10% |            10.3% |            0.1% |  -2.57 | 1.29 |     0.8761
   6 |   15% |            14.8% |            0.1% |  -2.80 | 1.43 |     0.8865
   7 |    8% |             8.1% |            0.0% |  -2.55 | 1.20 |     0.8831
```

100% of the saturation is at the **never-forget** rail. On block 0, 35% of
decay channels sit where `softplus'(f) = sigmoid(-4.1) ≈ 0.016` — the
gradient into `W_fdecay`/`gate_norm_f` is attenuated ~60×, so those channels
have lost *data-dependent* decay control. Their effective α now comes from
the static per-channel `gdn2_a` (which has trained up to `exp(a) ≈ 2.5` on
blk0, evidently compensating).

This is not currently a divergence — α values are healthy (0.87–0.96) and
the fraction has been flat, not growing, across the run. But it is exactly
the silent-freeze class that already cost this project two investigations
(β collapse → `beta_floor`; α clamp → `hi_frac`/`lo_frac` telemetry), and
right now it is **invisible**: the `[alpha]` line is dead under GDN-2
because the gdn2 branch never populates `_last_alpha_stats`.

**Fix (2 lines, `src/model.py` ~line 616):** populate
`self._last_alpha_stats = _alpha_stats(alpha_seq, lo=0.0, hi=1.0)` on the
gdn2 path, and split `decay_sat` into `decay_sat_lo` / `decay_sat_hi` in
`_gate_sat_stats`. Cheap, and it is the difference between noticing this
drift on day 2 of a 12-day run and finding it on day 12.

---

## B. Real bugs

### B1. `W_fdecay` is trained by AdamW; `W_b`/`W_w` are trained by Muon

`muon.py:195` — `is_gate = "gate" in name or "decay" in name`. The predicate
was written for `decay_raw` (1D) and `W_gate` (1×N). It also matches
`W_fdecay.weight`, which is a full (d_mem, n_neurons) matrix. Confirmed by
`temp/audit_routing.py`:

```
MUON  : blocks.N.W_b.weight (1152,2752), blocks.N.W_w.weight (1152,2752)
ADAMW : blocks.N.W_fdecay.weight (1152,2752)
```

So the three GDN-2 gates that the paper defines symmetrically are optimized
asymmetrically: erase/write get Muon at `muon_lr=5e-3` × muP scale 1.55 with
Newton-Schulz orthogonalization + NorMuon row-balancing; decay gets AdamW at
`lr=5e-4`. Roughly a 15× difference in step magnitude and a completely
different update geometry — on the one gate that controls forgetting, and
the one gate showing the A3 pinning.

**Do not change this before the flagship.** It is baked into arm B, W1–W5
and both P4 seeds; flipping it invalidates the recipe validation. Document
it, and treat it as suspect #1 if blk0's LO-saturation starts growing.
Worth its own 30M A/B afterwards (`is_gate` → `re.search(r'(^|\.)W_gate|decay_raw', name)`).

### B2. `src/energy.py` is stale — the headline efficiency number is inflated

`spiking_energy_per_token` predates both GDN-2 and hybrid attention:

- **The attention layer is not counted at all.** `HybridAttentionBlock` is
  `12·d²` MAC (qkv + o_proj + 4d FFN) plus `2·d·seq_len` for the attention
  itself. At d=1280 that is 19.7M MAC + 10.5M @ seq 4096 — completely
  missing from the report.
- **`ac = n_layers * spike_rate * N * dm * 3`** counts 3 spike-driven
  projections per block. GDN-2 made it 7 (`W_k, W_v, W_q, W_ff, W_b, W_w,
  W_fdecay`). AC is undercounted 2.33×.
- `n_layers` is applied to the spiking terms even though one of the 8/9
  layers is attention and emits no spikes.

Both errors push the same direction. Recomputing at the flagship shape
(spike 0.20, d=1280, L=9, 8 spiking + 1 attention):

| seq | reported ratio | corrected ratio |
|---|---|---|
| 512 | ~1.99× | **~1.6×** |
| 4096 | ~2.74× | **~2.1×** |

Still a genuine advantage — but the plan's "expect 2.7–2.9× at the
flagship" is ~30% optimistic, and the report's closing line ("spiking cost
is constant in seq_len") is now simply false: the attention layer's cost
grows with context exactly like a transformer's. This is one of the two
headline differentiators; it should be right before the flagship produces
the number that gets published. ~20 lines to fix.

### B3. `empirical_energy.py` is still unrunnable on any current checkpoint

P2 fixed its *builder* (`build_from_cfg` delegates correctly, `test_builders`
passes) but not its inputs:

- `SPK_DIR`/`TF_DIR` are hardcoded to `phase25_b` / `phase25_b_tf` — 20M-era
  runs. No `--run` argument. Same hardcoded-path bug that
  `brain_domain_transfer.py` was fixed for on 2026-08-07.
- `--compile-mode` **defaults to `reduce-overhead`** — the mode this project
  has documented as an open PyTorch bug and banned from every recipe.

So the "efficiency/sparsity" axis in Phase 5's success criteria has no
working measured instrument, only the (stale) analytical one from B2. Add
`--run` / `--tf-run`, flip the compile default to `default`, and add it to
`test_builders.py`.

### B4. `snapshot_brain` defaults **True** and is unbounded

`train_gpu.py:887` — `c.get("snapshot_brain", True)`. None of the W-arm or
P4 configs set it, so every eval writes a full brain to
`models/brains/<run>/` with no `keep_last` pruning. `p4_seed1` is already at
**1.1 GB / 31 snapshots**; `models/brains` totals 11 GB.

At the flagship: 213,623 steps ÷ `eval_every: 500` = **427 snapshots**, each
~59 MB at d_mem=1280/L=9 → **~25 GB**. The 500M v3 pre-launch audit already
caught this once and fixed it by setting `snapshot_brain: false`; the fix
never became the default, so it silently regressed for every run since.

**Set `snapshot_brain: false` in the flagship config** (and consider
flipping the default).

---

## C. Methodology

### C1. The LR cosine must be allowed to finish

Every screen to date runs `max_tokens: 1.1B` truncated by a wall-clock
`timeout`, so they stop mid-schedule:

```
step  4000 (131.1M tok): lr_scale=0.977
step 15600 (511.2M tok): lr_scale=0.615    <- p4_seed1 is here
step 33569 (1100.0M tok): lr_scale=0.100
```

That is fine and fair for A/B screens — all arms stop at the same scale.
It is **not** fine for the flagship, whose whole point is to be a serious
artifact: a model stopped at 60% of peak LR is materially worse than the
same model annealed to `lr_min_frac`. Set `max_tokens` to the budget you
will actually run, and either drop the wall-clock `timeout` or size it
generously enough that the cosine completes. If it has to be cut short,
cut it short *knowingly* and run a short anneal-to-zero tail.

### C2. Within-batch data diversity is 1 region, not `batch_size`

`StreamText.batch` does one sequential read of `B*(T+1)*4` tokens and then
draws all B windows from inside that single ~33k-token span. Literal token
overlap is low (~3%), but all 16 windows in a micro-batch come from the same
document neighbourhood — the *effective* number of independent contexts per
micro-batch is ~1, not 16.

This is worth naming because it reframes **W4**: `grad_accum: 4` didn't only
quadruple the effective batch, it quadrupled the number of independent
corpus regions per optimizer step. If region diversity is what W4 actually
bought, then drawing B independent starts across the corpus is the same win
at zero extra tokens — and it is a ~5-line change (B random `memmap` reads
of ~2 KB each; page-cache-friendly, negligible I/O).

Not a pre-flagship change — but a cheap, high-information 30M/90M A/B for
the idle 5080 while P4 finishes, and directly relevant to whether the
flagship should run grad_accum 4 or something else.

### C3. `recall_eval` at B=64 is noisy at exactly the N where the decision is made

`recall()` defaults to `B=64` with a fixed `seed=0`. At the observed
p≈0.41 (N=64), the binomial SE is **6.1 points** — so the 2-seed gate
("spread ≤15%") is partly measuring sampling noise at the high-N end, which
is precisely where W5's win over arm B (41% vs 27%) lives. The fixed seed
makes comparisons paired, which helps, but the CI is still wide.

**Run the P4 gate with `--B 256`.** It is ~4× a 3-minute eval and it halves
the error bar. Same for the flagship's recall baseline.

### C4. `--replay split` structurally disables the attention layer

Pass 2 (`model(x[:, -1:], initial_state=state)`) carries `M`/`mem`/`prev_spk`
but attention has no KV cache — `HybridAttentionBlock` is stateless by
design. So in pass 2 the attention layer sees a single token and contributes
nothing but its FFN.

This is consistent with the project's framing (split-replay is *meant* to
isolate persistent memory) and the Phase-1 3:1 finding was independently
confirmed without split-replay, so no conclusion changes. But it should be
stated explicitly in the finding: under split-replay, "more attention
layers" also means "more of the stack is inert in pass 2", so the metric is
not neutral with respect to attention ratio. Relevant when the 1B revisits
the ratio.

### C5. MQAR covers 43% of the vocabulary

`MQAR_{KEY,VAL}_{LO,HI}` in `train_gpu.py:45` and `KEY/VAL_{LO,HI}` in
`recall_eval.py:32` are both the 16k-vocab band (keys 1000–6000, values
8000–14000). Train and eval agree, so no result is invalid — but at
`vocab: 32768`, ids 14000–32767 are never exercised by the recall
augmentation or the recall metric. Already on the polish backlog; noting it
here because the flagship's recall number is the project's headline claim
and it is measured on 43% of the token space.

---

## D. Minor / cosmetic

| # | item | where |
|---|---|---|
| D1 | Attention block uses a **GELU MLP, not SwiGLU** — `use_swiglu` is unset in every current config, so `HybridAttentionBlock` takes the `nn.Sequential(Linear, GELU, Linear)` branch. Docs/design consistently say "RoPE + SwiGLU". | `hybrid_attn.py:124` |
| D2 | `depth_scaled_init` unset → `depth_scale = 1.0` at L=9. Standard practice would scale residual-adjacent output projections by `1/sqrt(2L)`. Not a bug; a knob that was never screened and gets slightly more relevant at depth 9. | `model.py:892` |
| D3 | L=9 with attention at `[4]` is **8:1**, not 7:1 as the plan/polish docs state. Direction is safe (Phase 1 showed *more* attention hurts persistent recall), but the ratio should be named correctly. | `plan_2026_08_05.md` §Phase 5 |
| D4 | `W_beta`/`W_alpha` are still built under gdn2 (49.2k params at 363M). Confirmed genuinely inert — they receive no gradient, so both optimizers skip them via `p.grad is None`. Purely state_dict/diagnostic noise. | `model.py:250,338` |
| D5 | `mtp_offset: 1` would slice `mtp_logits[:, :0]` (empty) and silently contribute a NaN/degenerate loss. All configs use 2. Latent footgun, one `assert` away. | `train_gpu.py:685` |
| D6 | The `initial_state` path hardcodes `_bcast(..., (B, d_mem, d_mem))` instead of reading `blk.n_state_units`/`blk.d_state`. It broadcasts the attention block's (B,1,1) dummy state into a full (B,1152,1152) tensor — ~340 MB at `recall_eval --B 64`. Wasteful, not wrong today; would be **wrong** if `n_heads > 1` were ever combined with `initial_state`. | `model.py:1159` |
| D7 | `eval_every: 500` → 427 evals on the flagship. Measured cost ≈ 63 s/eval at 276M (from the tok/s dip in the eval window), ~80 s at 363M → **~9.5 h**, because `evaluate()` runs `raw_model` eager by design. `eval_every: 1000` recovers ~5 h and still gives 214 eval points. | `train_gpu.py:756` |
| D8 | `W_fdecay.bias.fill_(-3.0)` is commented as "matches default lam≈0.98". It actually gives α ≈ 0.953 (half-life 14 tokens, not 34). Harmless — it's a trainable init — but the comment is wrong. | `model.py:283` |

---

## E. What was checked and is clean

- **Telemetry is not polluted by the ablated pass.** `evaluate()`'s last
  forward is `ppl(True)`, but `ablate_memory=True` skips the whole
  `if not ablate_memory:` block, so `_last_beta_stats` / `_last_gate_stats`
  still reflect the preceding non-ablated forward. No bug.
- **Train/val split is clean.** Train windows end at `0.999·len`; val starts
  there. `region` never crosses the boundary in either direction.
- **Corpus coverage.** 7B tokens over 14.5B with ~2× region oversampling —
  ~48% of tokens consumed, no meaningful repetition. Plan's claim holds.
- **MTP alignment is correct** at `offset=2` (`mtp[:, :-1]` vs `y[:, 1:]`
  predicts t+2).
- **Sparsity gradient path is live** — `spk` carries the atan surrogate
  through `lif_parallel`'s straight-through, so the penalty reaches
  `to_current`.
- **`gdn2_chunked_dense` numerics** — autocast hard-disabled, two-level
  anchored gating keeps every exponential ≤ 1, `solve_triangular` instead of
  Neumann doubling. The three historical NaN paths are all closed. Cost
  scales linearly in T (LIF `_scan_chunked` caps its triangular block at
  512), so block_size changes don't blow up.
- **`test_builders.py` passes 6/6** — all P2 instrument scripts match the
  canonical build and load arm B with 0 missing / 0 unexpected.
- **Flagship shape verifies**: `d=1280, n=3072, d_mem=1280, L=9,
  attention_heads=10` → **363.54M params**, head_dim 128, `d_mem % heads == 0`.
  Matches the plan.
- **NaN guard, watchdogs, SIGINT handler, `-k 300`, atomic ckpt rename,
  pinned-buffer reuse, resume RNG advance** — all present and correct.

---

## F. Suggested pre-launch checklist

**Decide (Glenn):**
1. Token budget vs wall clock — 7B/12.5d, or 4.5–5.5B/8–10d (A1).
2. `block_size: 512` locked (A2) — recommend yes.

**Land before launch (all small):**
3. `snapshot_brain: false` in the flagship config (B4).
4. Signed decay-gate telemetry + restore `[alpha]` under GDN-2 (A3).
5. `max_tokens` = real budget; no truncating `timeout` (C1).
6. `eval_every: 1000` (D7).
7. Fix `energy.py` before it produces the headline number (B2).
8. `empirical_energy.py`: `--run` arg + compile default off `reduce-overhead` (B3).

**Run the gate properly:**
9. P4 2-seed recall gate at `--B 256`, not 64 (C3).

**Document, don't change:**
10. `W_fdecay` → AdamW asymmetry (B1). Suspect #1 if blk0 LO-saturation grows.

**Free 5080 work while P4 finishes:**
11. 30M A/B on independent-start batch sampling (C2).
