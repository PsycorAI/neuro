# Next experiments after r25w / r25y / way 2 (2026-10-06, Neuro 52)

Plan for the tests beyond the running (r25w, r25y) and scheduled (way 2, v4_cloud.yaml) items, and how each
depends on their results. Deadline: v4 config freeze ~Oct 8, launch by ~Oct 11, demo Oct 19.

## 0. Where we stand
- Works: 12 attention layers + ONE spiking memory block at depth 6 + aligned chunked attention (r25o): synthetic
  recall beyond the window ~100%, +7.5% ppl vs r21b (inside Glenn's 8% budget). Conversion (r25t) = free, 5×.
- Open #1: real-text recall beyond the window. xcopy: memory 27.5 vs memory-off 27.3 (r25v); NL facts at chance.
- v4 is 13L at d 2048 (`configs/v4_cloud_bench.yaml`), the same depth as r25o. So the depth risk from G3 does not
  apply; the risk is width (stronger attention) and the 16B-token schedule.
- **Rejected 10-06 (CPU check, `temp/doclen_xchunk_check.py`):** "the corpus gives memory no cross-chunk job". In a
  T1024 window, ~76% of chunk-2 tokens continue a document from chunk 1 (old part and fineweb-edu alike).

## 1. Eval-only diagnostics (minutes each, no training; ckpts r25o, r25v, r25w)
These decide the real-text recall branch below. Run between 5090 runs, or on the 5080 before 18:00.

| id | test | what it tells us |
|---|---|---|
| E1 | xcopy sweep: span ends 0 / 64 / 256 / 480 tokens before chunk B; span length 8 and 32; real-text spans vs random-token spans | gain only near the boundary = **interference** (too many writes); random spans work but real ones don't = **selectivity** (memory writes only MQAR-like tokens); nothing works even at the boundary = **read path** |
| E2 | write-strength probe: β and memory-read norm on span tokens vs filler vs MQAR key/value tokens | confirms E1's cause |
| E3 | multi-chunk retention: fact in chunk 1, query in chunk 2 / 3 / 4 / 8 (T up to 4096) + per-chunk ppl at T4096 | the demo chat runs past 1024 tokens, but training sees only 1 boundary. If recall decays across chunks → add a T2048 final phase to v4 (idea H) |
| E4 | honest energy v2 (CPU): add a weight-fetch (SRAM read) cost per spike-triggered access; report prompt vs generated tokens (the LM head runs only on generated tokens) | keeps the demo's energy meter honest; the current 45 nm MAC/AC numbers ignore data movement |

### Results 10-06 18:15 (5080; `logs/e123_diag.log`; scripts `temp/xcopy_sweep_eval.py`, `temp/mem_write_probe.py`,
`temp/multichunk_eval.py`, `temp/xcopy_random_inwin.py`; runner `temp/run_e123.sh`)
- **Cause = SELECTIVITY.** The write gate is 0.83–0.91 on MQAR key/value tokens but sits at the floor (0.10 =
  `beta_floor`) on all ordinary text, real or random (r25o, r25v, r25w alike). The memory has learned an "MQAR
  detector", not a general write policy.
- Key matching is not the problem: for random spans the right write is the top-scoring one 95–100% of the time. Yet
  random-span copy beyond the window is 0.0%, against 50–72% inside a chunk (attention).
- Real-text copy: after subtracting a no-repeat control, the memory adds ≤3 points when the span ends at the boundary
  and ~0 at 448 tokens back. That is a small recency effect, not recall.
- xcopy augmentation (r25v) did not open the gate (0.116 vs 0.117) and it **hurt retention**: N8 haystack 28% at chunk 8.
- E3 = pass for r25o / r25w: N8 91–97% and N32 ~88% at chunk 8 (T4096), and memory lowers ppl in every chunk. No
  T2048 phase needed for retention. Minor: chunk 8 ppl is ~1 point worse than chunk 1, with memory on or off (a
  position effect?). Check that demo streaming resets positions per chunk.
- So the real-text branch is **selectivity**: phrase / scattered MQAR (p26e) and surprise-gated writes (p26d) come
  first; a bigger state (p26c) and later-layer reads (p26f) are dropped for now.

### p26e launched 10-06 18:51 (5080, Glenn OK; `temp/queue_5080_p26e.sh`)
- New flag `mqar_xchunk_scatter` (default 0 = byte-identical, RNG untouched; `src/train_gpu.py` `mqar_scatter_batch`):
  with this probability a cross-chunk MQAR batch scatters N whole-vocab key→value pairs at random pair slots through
  chunk A's real text, instead of the range-coded block at its start. CPU test `temp/test_mqar_scatter.py`; GPU smoke OK.
- `screen_p26e_7L_xscatter` = p25o + scatter 1.0 (`temp/mk_p26_configs.py`). Glenn needs the 5080 right after
  training, so the evals are DEFERRED: `temp/wait_p26e_end.sh` logs the end and marks DONE; afterwards run
  `GPU=1 bash temp/eval_p26e_deferred.sh` (recall, haystack, r25 evals, E1–E3 + random-span control vs p25o).
- Success = E2 write gate rises on ordinary tokens + E1 random-span copy beyond the window ≫ 0, at ≤ +3% chunk-1 ppl vs
  p25o (69.0) with block haystack kept. Then: real-text E1 gain, NL facts, and a 12L run on the 5090.
- p26d (surprise-gated writes) not built yet: the gate already sees the hidden state, so the missing piece is likely the
  training signal (p26e). Build p26d only if p26e opens the gate on random tokens but real-text recall stays at chance.

### p26e result 10-06 22:45 (vs control p25o, 7L)
- Write gate opens on unusual tokens (random spans 0.30 vs 0.10; text 0.15 vs 0.10). Real-text copy gain beyond the
  no-repeat baseline near the boundary: **+8 pts vs +2.5** (S12 d0), +5.8 vs +2.1 at d64, gone by d256.
- **Costs too much:** val_ppl 77.0 vs 66.0 (+17%; gap opens when xchunk starts at 30%); in-window recall 69% vs 97%;
  block cross-chunk haystack N8 41–73% vs ~91%; multi-chunk retention collapses (N8 9% at chunk 8). NL facts at chance.
- The random-span control is uninformative at 7L: even in-window, random-span copy is < 1% (attention can't copy them).
- → dose down (p26g, scatter 0.25) and make the pairs text-like (p26h, phrase mode: 2-token keys of ordinary words).

### Overnight queue 10-06 23:00 (Glenn: keep both GPUs busy)
- 5080 (`temp/queue_5080_p26.sh`, run + r25 evals + E1–E3 each): **p26g** scatter 0.25 → **p26h** phrase 0.25
  (`mqar_scatter_mode: phrase`, `mqar_phrase_batch`; CPU tests `temp/test_mqar_phrase.py`, `temp/smoke_phrase_cpu.py`).
- 5090 (`temp/queue_5090_n53.sh`, starts after Neuro 52's r25z queue): **r25x** = r25w conversion at threshold
  scale 1.0 (does q/v + KD give ~6× at ~0%?) → **r26c** = r25o with chunk 1024 / T2048 (longer in-window context
  for the demo chat).

### Glenn's idea (10-06 23:10): accept the memory ppl cost in pretraining, recover it at the end
Precedent: spike conversion lost ~50% ppl zero-shot and recovered all of it in a short phase with self-KD (r25t).
Test (5080, after p26h): POLISH phases from the p26e ckpt, +25M tok, scatter dose 1.0 → 0.25 as rehearsal
(`temp/mk_polish_seed.py` seeds the run; the trainer continues the schedule):
- **p26m** = anneal + low-dose rehearsal only; **p26n** = + KD from p25o (better ppl, same layout).
- Win = ppl back near p25o (66) while E1/E2 keep p26e's memory gains (gate on unusual tokens, real-text copy near the
  boundary). At 1B there is no twin teacher: p26m is the realistic arm; p26n tells us what a teacher adds (an external
  teacher such as the existing Llama-3.2-3B KD path would play that role).

### Overnight results (10-07, Neuro 53; all 7L vs p25o unless noted; `logs/gap_queue.log` [n53] lines)
| run | what | val_ppl | memory / energy |
|---|---|---|---|
| p25o / p26s | control seed 0 / 1 | 66.04 / 64.36 | 7L seed noise ~2.5% |
| p26g | scatter 0.25 | 72.76 (+10%) | synthetic recall kept; real-text copy gain ~+2 pts over p25o |
| p26h | phrase keys 0.25 | 72.19 (+9%) | same as p26g → rare tokens are NOT the cost |
| p26m / p26n | p26e + 25M polish (cosine tail) / + KD | 75.05 / 74.72 (p26e 77.03) | recovers 2–3 of 11 pts; memory gains kept |
| p26k / p26l | conversion 2× / 3× thresholds (40M) | chunk1 +24% / +60% | 5.1× / 7.4× honest (SRAM) |
| **r25x** (12L) | r25w recipe at 1.0× thresholds | chunk1 41.75 = dense | 5.1× compute, memory intact |
| r25z (12L, Neuro 52) | way 2: memory at 4 + 9 | chunk1 48.45 (+16%) | cross-chunk CHANCE → closed-neg |
| p26q / p26r | rewarmed polish (WSD 0.3×) / + KD | 76.44 / 75.50 | no better than the tail polish |
| p26v | rewarmed polish, scatter OFF | 75.75 | cost is in the weights; memory gains persist → scatter NOT adopted |
| p26u | 2× thresholds, 120M conversion | chunk1 72.27 (+4.7%) | 8.5× / 4.35× SRAM: learned thresholds drift down → p26x (frozen) |
| **r26c** (12L) | r25o at chunk 1024 / T2048 | chunk1 **40.27** (r25o 41.74; r21b +3.7%) | cross-chunk N8 ~100%, in-window NL K1@200 62% vs 41%; 14–21% slower at 150M |
- **v4 recommendation so far:** r26c layout (chunk 1024 / T2048) + r25x-style conversion (no ppl cost, ~5× compute /
  ~3× honest), thresholds chosen last from the dense ckpt (p26u sizes the cost); scatter MQAR NOT adopted (ppl cost
  does not polish away at 7L; p26v = polish with scatter off still pending). Re-bench tok/s on the H200.
- Real-text recall: every scatter variant costs ~9–17% ppl at 7L for +2–4 pts of copy beyond the window → poor trade.
- Polish at the cosine tail barely moves ppl → rewarmed polish p26q / p26r (WSD from the resume step, 0.3× peak).
- Energy: honest > ~4× costs too much at 40M conversion tokens → p26u (2×, 120M) sizes the v4 conversion phase.

### RELEASE CANDIDATE + weakness tests (10-07 09:50, Glenn: RC first, then strengthen ppl / energy / throughput)
**RC** = r26c layout (12 attention + one spiking memory block at depth 6, chunk 1024 / T2048, xchunk 0.5 from 30%) +
r25x conversion (L8 spike codes on every linear input + q/v, self-KD, 80M tok, thresholds 1.0×); keep the dense ckpt
and choose thresholds last. Weak spots: ppl (+3.7% vs r21b), honest energy (~3×), throughput (14–21% slower at 150M).
| weakness | test | where / when |
|---|---|---|
| (RC itself) | r26t = r26c seed 1; **r26d** = RC conversion on r26c + honest energy + retention | 5090 → ~13:00 / ~16:00 |
| throughput | `temp/bench_rc_tps.sh`: B4×GA2 vs B8×GA1 vs `mqar_once_per_step` (same tokens/step) | 5090 ~13:00 (40 min) |
| ppl | **r26e** = r26c with xchunk 0.25 from 50% (smaller aux bump) | 5090 after r26d |
| energy | `temp/site_sens_eval.py` (per-site zero-shot sensitivity, 7L + 12L); **p26w** = thresholds 2× only at head + MLP input (`SITE_SCALE` in `temp/mk_spike_seed.py`) | 5080 ~12:30–13:45 |
| energy | p26u = 2× thresholds, 120M conversion tokens (does longer conversion shrink the cost?) | 5080 → ~11:40 |
| ppl / memory | p26v = polish with scatter OFF | 5080 → ~12:30 |

### RC review + improvement search (10-07 13:30, Glenn: review results / RC trade-offs, search, iterate)
**RC strengths:** synthetic beyond-window recall ~100% and retention to chunk 8; in-window NL recall improved by chunk
1024; free conversion (no ppl cost, ~5× compute / ~3× honest); ppl +3.7% vs attention-only.
**RC weaknesses and what each test says so far:**
1. *Real-text memory* (main gap). Cause = the write gate learned an "MQAR detector" (E2). Synthetic fixes (scatter, phrase,
   xcopy) cost 9–17% ppl at 7L that no polish recovers. **New idea (lit: SWAX, ICLR 2026): stochastic chunk size** — the
   plain LM loss on real text then trains the memory. Built (`attn_chunk_short`); p26y (7L) / r26f (12L) queued.
2. *ppl* (+3.7%): the xchunk aux start bump (r26c 72.9 → 82.4 at 30%) → r26e (dose 0.25 from 50%). Stochastic chunks
   may also replace part of the xchunk aux. Idea: **external-teacher KD in the conversion phase** (we already pay KD
   cost there; a Llama-3.2-3B teacher could push ppl below the dense model) — to-test, check `src/kd.py` LUT path.
3. *Energy*: thresholds are the lever; per-site raising is not a free lunch (~1 µJ per 1% ppl everywhere); longer
   conversion recovers ppl but thresholds drift down → p26x (frozen). Accounting idea: the LM head (largest site) is
   needed only for generated tokens, not the prompt → report prefill vs generation energy in the demo (to-build).
4. *Throughput* (−15–20% vs chunk 512): bench running (B8×GA1, `mqar_once_per_step`); stochastic chunks speed up the
   short steps (128-token attention).
**Bigger ideas for v4.x (post-launch):** HOLA (2607.02303: exact bounded cache of high-residual tokens next to the
linear state, +16% ppl at 340M, strong NIAH) = the "episodic store" idea with a recipe; ARMT / RMT memory tokens
(2407.04841: 79.9% single-fact QA at 50M tokens on BABILong) for real-text facts beyond the window.

### Hippocampus (HOLA) — research + build (10-07 15:45, Glenn: 1st choice = in v4 from the start)
**Seven open-source "hippocampus" projects, what we take (neuromorphic-compatible only):**
- (a) **AHN** (ByteDance, 2510.07318): GDN memory written as tokens LEAVE the attention window (= our chunks);
  randomized window sizes; trained by KL self-distillation from a full-attention teacher (denser signal than CE). TAKE:
  random chunks (built), teacher KD (r26g doubles as this test). Training-only, so neuromorphic-neutral.
- (b) **Hippoformer** (ICLR 2026): grid-cell spatial memory for 2D/3D prediction — not useful for an LM. Skip.
- (c) **TEM / TEM-t**: memory = Hopfield/attention binding a structural code ("where") with content ("what"). Idea
  (parked): add a relative-time/position code to the HOLA cache keys.
- (d) **SNN hippocampus (CA3 STDP, sPyMem on SpiNNaker)**: one-shot Hebbian storage, NOVELTY-gated writes, pattern
  completion, replay. Supports HOLA's surprise-scored cache; ideas (to-test later): winner-take-all (top-1/top-k) cache
  read at inference = spike-friendly; "sleep" replay of evicted entries into the delta-rule state (consolidation).
- (e) HippocampAI, (f) MemGPT/Letta, (g) HippoRAG: agent-level external stores (databases, knowledge graphs,
  PageRank) — not in-model, not neuromorphic. Only demo-UX ideas (memory strength / decay display). Skip for v4.
**Build (flag `hola_w`, default 0 = byte-identical):** `SpikingHebbianBlock.enable_hola` / `_hola_read` in `src/model.py`;
`gdn2_chunked_dense(..., want_u=True)` in `src/delta_chunked.py` returns |u_t| (the committed delta-rule correction =
HOLA's β·‖e‖ eviction score). At each attention-chunk boundary the cache keeps the top-64 earlier tokens; queries read
them by RMSNorm-γ-sharpened softmax + null sink; added to the state read through a zero-init per-channel λ.
CPU tests `temp/test_hola_cpu.py`: kernel bit-identical with want_u, causal with HOLA on (0.0 change before a perturbed
token), cache live, gradients reach all HOLA params. Screens: **p26z** (7L), **p26za** (7L + random chunks), **r26h**
(12L RC + HOLA). Throughput note: B8×GA1 and `mqar_once_per_step` both spilled the 5090 (38–39 GB reserved while
only 3–4 GB allocated) → re-bench with `expandable_segments` (to-test).

## 2. What decides v4 (5090, by Oct 8)
1. **r25w / r25y (running)**: pick the conversion recipe and head rank. Rule: chunk-1 ppl ≤ +2% vs r25o, cross-chunk
   haystack ≥ 90%, energy ≥ 5×. If r25w's 1.5× thresholds cost > 2%, keep the r25t recipe plus q/v coding.
2. **Way 2 (scheduled)**, r26a = 14L, memory at depths 4 + 9.
   - Adopt for v4 if cross-chunk haystack N32 beats r25o (84–100) OR xcopy / NL facts beyond the window rise above
     memory-off, at ≤ +3% ppl vs r25o.
   - Otherwise v4 stays 13L (one memory block).
3. **Cloud rehearsal (5080, 7L)**: run the exact v4 pipeline end to end at small scale (pretrain → `mk_spike_seed` →
   conversion → evals) with `configs/v4_cloud.yaml` scaled down. The cloud run cannot afford pipeline bugs.
4. **xchunk start 0.1 vs 0.3 (5080, 7L)**: r25o starts xchunk MQAR at 30% of training, which is ~5B tokens (1.5 days)
   into v4 before memory can be judged. If 0.1 costs no ppl, use it, so the go/no-go comes at ~2B tokens.

## 3. Real-text recall (branch on E1/E2; 7L on the 5080 first, then one 12L run on the 5090)
| E1/E2 result | next test |
|---|---|
| interference | p26c: memory state 2× (d_mem or more key dims); p26d: surprise-gated writes (β from prediction error, idea G). Way 2 also doubles the state. |
| selectivity | p26e: phrase-MQAR, with multi-token keys and values cut from real text and placed inside real text (a bridge from synthetic to NL); NL-fact aux from the xchunk start |
| read path | p26f: memory read at later layers (`mem_read_layers` [9, 12]) with contextual queries; multi-reader (idea E) |

- Width note: v4's memory state is ~7× r25o's (d 2048 vs 768). If the cause is capacity, v4 may partly fix it on its
  own, so add xcopy + NL-fact evals to the cloud ckpt evals.
- Post-launch (v4.x), if all of the above fail: an episodic store of salient tokens (Titans-MAC / memorizing-transformer
  style top-k store, spike-gated writes) = a real hippocampus. Too big to build before the freeze.

## 4. More energy (5080, 7L; adopt into the v4 conversion phase only if free)
- p26g per-site thresholds: LM-head input 2×, the rest 1.5× (the head is 30–40% of the remaining energy).
- p26h spike-count penalty (L1 on |s|) instead of a fixed threshold scale.
- p26i code k as well as q, v (QK^T fully spike-driven).
- p26j drop the attention output gate (6.7 µJ/token; needs training from scratch).

## 5. Only if a 5090 slot opens: the +7.5% chunking cost
r26k = r25o without xchunk MQAR (separates the chunking cost from the aux-task cost). Low priority: within budget.

## 6. Cloud run monitoring (into `configs/v4_cloud.yaml` / run scripts)
- Ckpt evals every ~1B tokens on the instance: chunk-1/2 ppl, memory Δ, cross-chunk haystack, xcopy, NL facts.
- Go/no-go ~1B tokens after the xchunk start: haystack N8 > 50% and memory Δ on chunk 2 > 0. If not, continue as an
  attention model (memory stays a streaming bonus) and tell Glenn.
- Keep the dense ckpt before conversion as the fallback demo model.
- Re-bench the final layout on the H200 (T1024 / chunk 512 / B32) before launch; the 10-04 bench used the old layout.

## 7. GPU sketch (approximate)
- **5090:** r25w (~17:35) → r25y 256 (~19:15) → r25y 512 (~21:00) → r26a way 2 (~02:00 Oct 7) → E1–E3 (~30 min) →
  one 12L real-text test picked by E1 (Oct 7 day) → freeze Oct 8.
- **5080:** Glenn's 18:00–24:00. Oct 7 00:00–19:00: cloud rehearsal → xchunk-start test → 2 real-text 7L tests →
  2 energy 7L tests.
- Seeds: r25o is a single seed; the 12L run on Oct 7 doubles as a 2nd seed of the layout if it keeps the recipe.
