# Contiguous (stateful) training + "memory must carry" — design doc

2026-09-29. Scoped via /new-feature. Status: **BUILT + DEPLOYED (flag-off); SCREENED 2026-09-30 → CLOSED-NEG at 30M on the
signed stack.**

**Outcome (2026-09-30):**
- r9a stateful: zero-state val 70.32, streaming 65.3, haystack N8 100/95/67/35, N32 100/89/64/23.
- r9b stochastic window: 61.19, streaming 64.3, i.e. the same as control.
- r9c both: 70.49, streaming 65.2, the same as r9a.
- Signed control (2 seeds): 61.24, streaming 64.3.
- Every primary criterion failed. Stateful runs got modest N32 haystack gains (+8–14 pt at gaps 64/192) but no gap-440 gain.
- 1M-token streams (`temp/stream_eval_long.py`): signed random-trained models are already length-stable (flat ‖M‖). Only the
  gap440 model explodes (‖M‖ 27.7→75.5, ppl +13%). So state passing has nothing to fix for signed models at this scale, but it
  *is* the fix if a gap-curriculum recipe is ever adopted.
- Side findings:
  - Short-window inference costs only ~1% on signed models, which makes constant-memory inference viable
    (`temp/eval_attn_window.py`).
  - The stateful runs trained 2.2× faster. That exposed the 8 MB read-ahead data-loader bottleneck (fix staged in
    `temp/stage_madv`).

## 1. What it is

Train the model the way it is used: as a **continuous stream**. Each of the B batch rows is a read cursor that moves
through the corpus in order, one 512-token window per step, and the GDN memory state `M` (plus the boundary spike
`prev_spk`) left at the end of window t becomes the initial state of window t+1 for the same row (truncated BPTT:
state carried forward, gradient stopped at the window boundary). Compute per token is unchanged.

It comes with a companion change, because the diagnosis below shows the two are needed together. The hybrid's
attention layer gets a **stochastic sliding window** (short or long, chosen per step; SWAX recipe), so
anything beyond the window must come through the memory.

It is **not**: full BPTT across windows; Transformer-XL-style attention caching (attention stays window-local); a
data re-ordering project (related-document packing is Phase 3, deferred); or a change to the LIF neuron (its
membrane still restarts at 0 each window, see §3).

## 2. Why now (evidence gathered 2026-09-29)

**(a) The recall-with-gaps failure is a memory-horizon problem, not an attention problem.** Measured with
`temp/diag_haystack_pathways.py` (`logs/diag_haystack_pathways.log`), haystack N8 accuracy at gap 0/64/192 and the
median per-channel half-life of the GDN decay gate on real text:

| model | acc gap 0/64/192 | mem_off | attn_off | median half-life (tok) |
|---|---|---|---|---|
| lif_exact (2 spiking) | 100 / 96 / 73 | chance | — | 47–76 |
| lev4 | 100 / 66 / 55 | chance | — | 38–43 |
| hybrid [spk, attn] | 100 / 14 / 14 | chance | 100 / 14 / 12 | **0–1** |
| full stack | 100 / 8 / 10 | chance | 100 / 7 / 12 | **1** |
| full + canon_a | 100 / 9 / 10 | chance | 100 / 11 / 11 | **1** |
| full + spk_mlp | 100 / 9 / 16 | chance | 100 / 7 / 13 | **0** |
| 3L [spk, attn, spk] | 100 / 31 / 18 | chance | 100 / 30 / 22 | 42–58 (2-block mean) |
| full + gap440 curriculum | 71 / 77 / 70 | chance | 75 / 79 / 72 | 16 (long channels retained) |

- Recall is done **entirely by the GDN memory** in every model. Switching attention off changes nothing, so the "attention
  shortcut" hypothesis is wrong.
- In the 2-layer hybrids the memory learned to **wipe most channels every token** on real text. Attention already
  supplies the local context the LM loss rewards, so nothing in pretraining rewards holding information. Every local
  mixer we add (attention, Canon-A, spk_mlp) makes this worse.
- The gap curriculum fixes this by rewarding retention with a synthetic task, at +25% ppl.

**(a2) Update 18:00 — signed spikes restore the horizon (1 seed).** `screen_r5_signed_s0_30m` (full stack + `lif_signed`)
= **61.36 val_ppl (0.90× the 2L tfmr)**, recall 100×5, haystack N8 100 / 88 / 65 / 34. Its memory's median half-life is
**97 / 55 tokens** (full stack: 1), and its streaming ppl is 64.6 (full stack: 80.2). The cost is 2.5× the spike events
(spiking-layer rate 0.264 vs 0.107). The mechanism is **unknown**. I tested one hypothesis — unsigned spike codes make the
memory keys overlap — and it was **NOT supported**: key |cos| is lif_exact 0.37, full 0.32, signed 0.25, gap440 0.42, which
doesn't track the horizon (`temp/diag_key_similarity.py`, `logs/diag_key_similarity.log`). **The r9 screens below are
therefore built on the signed stack.** Neuro 44 queued the signed follow-ups (r8: seed 1, `lam2` rate-matched sparsity,
canon, 3L).

**(b) Carried state already helps at eval — but long-retention models break without stateful training.**
`temp/stream_eval.py` (`logs/stream_eval_baseline.log`): 16 contiguous val streams × 36 windows, zero-state vs carried
state, all models trained on random windows from zero state.

| model | zero-state ppl | carried-state ppl | pos 0–15 zero→carry |
|---|---|---|---|
| lif_exact | 112.0 | 104.1 (−7.1%) | 208 → 111 |
| lev4 | 93.6 | 87.0 (−7.0%) | 172 → 95 |
| hybrid | 93.4 | 90.3 (−3.3%) | 158 → 111 |
| full stack | 83.5 | 80.2 (−4.0%) | 144 → 101 |
| full + gap440 | 103.9 | **128.7 (+23.9%)** | 168 → 155 |

- The cold-start positions are most of what carried state fixes. This is the same early-position gap found in the ppl-gap
  diagnosis (positions 4–15 were 1.72× the transformer).
- The gap-curriculum model **gets worse** with carried state. Its long-lived channels never saw a state older than 512
  tokens in training, so they never learned to clear stale content. This matches the "unexplored states hypothesis"
  (Buitrago & Gu, ICML 2025), and state passing / TBTT is their measured fix. So **longer memory without stateful
  training is unsafe for real streaming use**.

**(c) The data supports it.** `temp/diag_doc_lengths.py`: the pretraining stream is built from shuffled 8192-token
source segments. Documents are delimited by `<|begin_of_text|>` (maps to id 68 in the 16k vocab), so the gates can learn to
wipe on it. The median document is 298 tokens, but **87% of tokens are in documents longer than 512** (66% > 1k, 46% > 2k).
Most tokens therefore have in-document history that a 512 window cuts off today.

**(d) Why the June attempt was negative, and what's different now.** Journey §10 (June 20–21) tested it at 9M/22M, 80M
tokens, on a fixed-λ Hebbian memory (dead within ~500 tokens) and a home-built gated variant that lost on its own. Stateful
never beat random windows on absolute ppl. Differences now: (1) the substrate is GDN-2 (gated delta rule with
data-dependent decay), the class that works in the literature; (2) we now have evals that can see carried memory (haystack,
streaming by position), where June graded on training-length ppl only; (3) we now know **why** it matters: the hybrid's
memory horizon collapses without pressure. Document-boundary resets (June "Phase 6") were never tested.

**Differentiation (Glenn's goal).** A constant-memory model that is *trained* to live in a continuous stream, with local
attention for precision and the spiking memory for everything older. A full-attention hybrid loses the constant-memory
inference claim; a windowed one keeps it. Transformers pay a KV cost that grows with context to get the same history.

**Fit with priorities.** v3 post-training (5090) is unaffected. These are 30M screens for the 5080 (priority 2, v4
screens) and directly target the one open v4 blocker: recall under interference without the +25% ppl of the gap curriculum.

## 3. Where it plugs in (verified 2026-09-29)

Already exists (June, never used since):
- `src/data_stream.py:65-114` `SequentialStream`: B contiguous cursors, `next(T)` returns `(x, y, reset)`. No
  `skip()`, so a resume **replays data from the start** (bad with the 18:55 pause/resume).
- `src/train_gpu.py:455-471` enables it (`stateful_stream: true`); `:632` `carry_state`; `:661-691` stateful branch
  with optional `reset_prob` (stochastic resets).
- `src/model.py:1400` `forward(initial_state=..., return_final_state=True)`; `:1682-1686` final state is **detached**
  (TBPTT correct); `:878` GDN-2 chunked kernel takes `M` as init (carry works on the gdn2 path).
- `scripts/stateful_eval.py` (June): builds via canonical `build_model`, but its plot overwrites `assets/stateful_eval_curve.png`.
  Use `temp/stream_eval.py` instead.

Gaps that must be fixed before any screen:
1. **Stale stateful branch:** `train_gpu.py:661-691` computes plain CE only and `continue`s. It skips the **MQAR aux**
   (`:825-870`), fused CE, MTP and z-loss, so a stateful screen as-is would drop recall training (confound).
   Fix: the stateful branch should replace only the data fetch and forward, then fall through to the shared loss code.
2. **Resume replays data:** add `SequentialStream.skip(n_calls)` and call it after checkpoint resume (`:563-574`) with
   `step * grad_accum`. The carried state is lost at resume, which acts as one reset (acceptable).
3. **LIF membrane not carried:** `model.py:753`: `lif_parallel`/`lif_exact` are called without the incoming `mem`. Only
   `M` and `prev_spk` carry. The membrane time constant is a few tokens, so this is minor. Documented, not fixed in v1
   (same as the known KV-decode gap in `neuro_kv_cache_not_decomposable`).
4. **Attention window:** `src/hybrid_attn.py:145-164` `_attn` uses `F.scaled_dot_product_attention(is_causal=True)`
   (full 512). Add `attn_window` (int, 0 = full) with a causal band mask. The trainer picks the window each step
   with a **private RNG** (so data/MQAR draws stay byte-identical to the control). Construction is at
   `model.py:1187-1192`; config plumbing is `build_model` at `train_gpu.py:~177`.
   A band mask disables flash SDPA, which is fine at 30M. At 1B use flash-attn `window_size` or FlexAttention.

Eval (no src change): `temp/stream_eval.py` (carried vs zero by position), `temp/haystack_recall_eval.py`,
`temp/diag_haystack_pathways.py`, `scripts/recall_eval.py`. Add a **cross-window haystack** (facts in window t, query in
window t+k, carried state only). That is the direct test of "memory carries beyond attention's reach".

## 4. Cost & risk

- Compute: same FLOPs/token. `SequentialStream` reads B windows per step like `data_independent_windows` does. Memory: +1
  carried `M` per spiking layer (30M: 32×1024×1024 fp32 = 128 MB/layer).
- **Confound: data order.** Stateful rows see consecutive windows, so gradients are correlated across steps (with NorMuon
  momentum). The clean ablation is a third arm with *contiguous data but a reset every window*, which separates
  data order from state carry (June's 2×2 idea). Run it only if the stateful arm wins.
- **Failure mode: state blow-up on long streams** (Stuffed Mamba). Watch streaming ppl per window index (must not rise
  over 36 windows) and the carried `‖M‖`.
- **Failure mode: val_ppl is blind to it.** Standard val_ppl is zero-state and random-window, so a stateful model is
  graded on the one regime it wasn't trained for. That is what "failed" in June. Pre-register the streaming ppl and the
  haystack as the primary metrics, and val_ppl as a *no-regression* guard.
- Live-code risk: `src/` is imported by queued runs at start and by the 5090 post-training chain (~00:15 Wed). Deploy
  only via stage → `temp/parity_gap_flags.py` (`GAP_STAGE`, `PARITY_REF=/tmp/parity_5080_ref.pt`) → atomic `mv`.
- The window-choice RNG must not perturb the main RNG (a silent confound).

## 5. Implementation spec (ready to build)

Flags (all default off = byte-identical):
- `stateful_stream: true` (exists); `stateful_reset_prob` (reuse `reset_prob`, default 0; SP literature uses none).
- `attn_window: 0` (full). `attn_window_short: 64`, `attn_window_p: 0.5` (P(short) per step),
  `attn_window_long_frac: 0.1` (final 10% of steps always long; SWAX anneal). Eval uses the long window.

Code:
- `hybrid_attn.py`: `self.attn_window = 0` attribute; in `_attn`, if `0 < w < T`:
  `mask = (i >= j) & (i - j < w)` (T×T bool, cached per (T, w)), passed as `attn_mask` without `is_causal`.
- `train_gpu.py`: `_win_rng = np.random.RandomState(seed + 104729)`. Each step, if `attn_window_short`: pick short w.p.
  p unless `step >= (1 - long_frac) * total_steps`, then set `b.attn_window` on every attention block. Set back to
  long before `evaluate()`.
- `train_gpu.py` stateful: move `x, y, reset = seq_stream.next(...)` plus the reset handling into the data-fetch
  `if/else`. In the forward, pass `initial_state=carry_state, return_final_state=True` and unpack the extra state
  element. Delete the stateful `continue` so the shared loss (fused CE, MQAR aux, z-loss, MTP) applies.
- `data_stream.py`: `skip(n)` advances the cursors by `n*T` using the same wrap rule (no reads). Call it on resume.

## 6. Screens (30M, 100M tok, control = signed stack s0 61.36 + `screen_r8_signed_s1_30m`)

Configs: `temp/mk_gap_r9_configs.py` → `configs/screen_r9{a,b,c}_*`. **These need the staged code deployed first** —
live `src/` silently ignores `attn_window_*`, and its old stateful path skips the MQAR aux.

| arm | config | question |
|---|---|---|
| r9a | signed + `stateful_stream` | does training in the stream improve streaming ppl + gapped recall without hurting val_ppl? |
| r9b | signed + stochastic attn window (64 / full, p 0.5, last 10% full) | SWAX: does a short window force a still-longer memory at no ppl cost? |
| r9c | signed + stateful + stochastic window | the full "memory must carry" design |
| (later) | best r9 + canon_a / + gentle gap MQAR | after Neuro 44's r8 canon and the r7 gap results |
| (only if a stateful arm wins) | contiguous data, reset every window | data-order confound control |

## 7. Pre-registered success criteria (30M, seed 0; a win needs seed 1 before it counts)

Against the signed control (s0: val 61.36, carried streaming 64.6, haystack N8 gap64 88 / gap192 65 / gap440 34,
half-life 97/55), same seed, same 100M tokens:
1. **Primary: streaming ppl** (`temp/stream_eval.py`, carried) at least **4% better** than the control's 64.6 (≤ 62.0),
   and streaming ppl by window index must not rise from window 1 to window 35 (no blow-up).
2. **Primary: haystack at long gaps:** N8 gap 440 ≥ 50% (control 34) or N32 gap 192 ≥ 65% (control 48), with gap 0 ≥ 95%.
3. **Guard:** standard val_ppl within **+5%** of the control (≤ 64.4). Seed noise is ~7%, so a failure on this guard
   alone triggers a seed 1, not a kill.
4. **Guard:** immediate recall (recall_eval N16/32/64) ≥ 97/95/90.
5. Mechanism check: memory half-life on text not below the control's.

Kill: streaming ppl worse than the control's carried ppl, or val_ppl > +10% on both seeds.

## 8. Recommendation & decision gate

**Build now.** The code is ≈80 lines, flag-gated and staged in `temp/stage_r8` (not yet deployed). It passes 4 tests:
- model parity (`temp/parity_gap_flags.py`, ALL PASS)
- trainer-loop parity (`temp/test_r8_trainer.py`: 20-step losses live vs staged 0.048 max diff, live vs live 0.031 atomics noise; step 1 identical)
- unit tests (`temp/test_r8_units.py`: window mask vs manual reference 4e-7; `skip()` == `next()` exactly; zero-state == plain forward)
- a stateful resume smoke test

Run r9a–c after Neuro 44's signed seed 1 / lam2. Their evidence is stronger than the queued r5 leftovers, which were
built on the now-superseded full stack.
- If r9a or r9c passes: seed 1 → add to the ~100M honest mid-scale check (6–9 layers, attention-ratio sweep) → v4 recipe.
  Then post-training Phase 2: multi-turn SFT with state carried across turns, and brains/persistence evals (scorecard #23).
- If only r9b passes: adopt stochastic windowing in v4. Revisit stateful after the mid-scale check.
- If all fail the primary criteria: close with the substrate caveat removed (GDN-2 tested), and keep the gap curriculum
  as the recall fix. **Do not** ship a long-memory model without a streaming-ppl check (the gap440 +24% blow-up).

Phase 3 (deferred, needs results): In-Context-Pretraining-style ordering, with related documents consecutive in a stream
(kNN over document embeddings). Our 8192-token same-source segments are already a weak version of this.

## References (ranked)

Primary port targets
1. Buitrago Ruiz & Gu, *Understanding and Improving Length Generalization in Recurrent Models*, ICML 2025,
   [arXiv 2507.02782](https://arxiv.org/abs/2507.02782); [blog](https://goombalab.github.io/blog/2025/improving-length-generalization/).
   The "unexplored states hypothesis". State Passing (initial state = final state of the previous batch) and TBTT (contiguous
   chunks, gradient blocked) fixed length generalization in Mamba-2 up to 256k with ~500 post-training steps (~0.1% of
   pretraining). This is the recipe for r8a, and it explains our gap440 +24% streaming blow-up.
2. Cabannes et al., *Short window attention enables long-term memorization* (SWAX), ICLR 2026,
   [arXiv 2509.24552](https://arxiv.org/abs/2509.24552). SWA + xLSTM, 1:1 layers, 1.4B/7B: **larger windows hurt long
   context** because the RNN memory is undertrained. Stochastic window (128 vs 2048; P(short) 0.5 at 1.4B, 0.75 at 7B;
   final 10% fixed long) beat both fixed windows on short *and* long context (val loss 2.502 vs 2.523 for fixed-2048; RULER
   NIAH at 131k: ~30% for the 128 window vs ~0 for 2048). This is the recipe for r8b, and it independently matches our half-life diagnosis.

Eval rigor / failure modes
3. Chen et al., *Stuffed Mamba: Oversized States Lead to the Inability to Forget* ([arXiv 2410.07145](https://arxiv.org/abs/2410.07145)):
   state explosion beyond the training length. Hence the per-window streaming-ppl guard.
4. `logs/diag_haystack_pathways.log`, `logs/stream_eval_baseline.log` (this doc's baselines); journey §10 (June negative).

Lineage / related
5. Shi et al., *In-Context Pretraining* ([arXiv 2310.10638](https://arxiv.org/abs/2310.10638), ICLR 2024): related documents
   in one context improve in-context learning and reading comprehension (Phase 3).
6. StateLinFormer, *Stateful Training Enhancing Long-term Memory in Navigation* ([arXiv 2603.23571](https://arxiv.org/abs/2603.23571)):
   linear attention with the memory kept across training batches beats the stateless version and transformers on long interactions (non-LM).
7. Transformer-XL (Dai et al. 2019), Block-Recurrent Transformer (Hutchins et al. 2022), RMT (Bulatov et al. 2022):
   segment recurrence with cached or recurrent state across segments.
