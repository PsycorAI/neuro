# Memory that matters at depth — review, literature, ideas (2026-10-05, Neuro 51)

Glenn asked (10-04 late) for the best of both worlds: long attention **and** a long memory that visibly matters in the
v4 demo. This doc reviews what we tried, what the literature says, and the ideas now queued (round 25).

## 1. What we tried (memory at depth)

| run | shape | change | val_ppl | memory Δ | haystack beyond reach | verdict |
|---|---|---|---|---|---|---|
| G3 cand | 150M, 12L, 1B tok | H0 + 11 × attn | 27.26 | ≤ 0, β 97% at floor | — | memory switched off by the LM loss |
| g3_h2 | same | 2nd H at layer 8 | (cancelled) | +13 → +1.5 | — | fades the same way |
| r21a | 150M, 200M tok | control | 36.46 | ~0 | — | |
| r21c | | window 128 + late gap dose | 40.82 (+12%) | 0.00 | 14% (chance) | attention relays across windows; memory never learns |
| r21g | | window 32 + gap dose from step 0 | 48.45 (+33%) | small | 57% N8 | memory *can* learn when relay is impossible, at a big cost |
| r21i | | full window + gap dose from step 0 | 43.32 (+19%) | +0.1 | in-window 100% via **attention** | attention learns any in-context job |
| r23a | 4L, G3 width | depth proxy | 77.6 | **+6.9** | — | fewer layers → memory used: depth is a cause |
| r24a | 12L | SWAX window 32/full | running | +8 → +0.5 by eval 4, β 99% at floor | — | looks like the same failure |
| r24b | 12L | stateful segments (detached) | queued | | | |
| r9 (30M) | 2L signed | stateful segments | closed-neg | | +8–14 pt N32 | natural text rewards carried state little at 30M |

Baseline of the new evals on r21i, with attention forced into 512-token chunks (`temp/run_r25_evals.sh`):
chunk-2 memory Δ +0.15 ppl (+0.46 on the first 64 tokens), cross-chunk haystack at chance, NL facts beyond the window
at chance (6.8%; chance 8.3%). Even **inside** the window, NL fact recall with 200 filler tokens is only 28%.

## 2. Root causes (synthesis)

1. **Competition.** Attention solves every in-context job faster, so the gradient feeds it and the memory's write gate
   collapses (β to floor). Any route attention has, it takes (r21i: it even learned 440-token haystack).
2. **Relay.** A sliding window does not block attention at depth: reach is L × W (11 × 128 ≈ 1,400 > 512). Only W
   below T/L blocks it (r21g, w32), which costs +33% ppl.
3. **Write-path starvation.** With detached state (TBPTT, r9/r24b) the write projections only get gradient from
   reads inside the same segment — where attention already wins.
4. **Token-level keys.** H0 sits at layer 0: its keys come from the previous token's spikes (an induction store). It can
   do MQAR and "A B … A → B" copying, but a paraphrased question ("Where does Alice live?") needs contextual keys.
5. **Data.** At this scale natural text rarely needs more than 512 tokens of memory (r9), so the LM loss alone barely
   rewards long memory; the demo skill (fact → question later) is not in the training signal at all.

## 3. Literature (short)

- **SWAX** (arXiv 2509.24552): short sliding windows force the recurrent layers to learn; stochastic windows keep
  short-context quality. Our r21c/r24a show the relay loophole at 12 layers.
- **"Some attention is all you need for retrieval"** (2025): in RecurrentGemma and Jamba, retrieval runs *only* through
  attention; ablating attention gives 0%. Same as our G3.
- **RMT / ARMT** (arXiv 2207.06881, 2407.04841): attention inside a segment, memory carried across segments, trained
  with **BPTT across segments** and a segment curriculum. ARMT adds a layerwise associative (delta-rule) memory and
  answers BABILong single-fact questions over 50M tokens (79.9%). Recipe: 512-token segments, full BPTT, a segment
  curriculum 2 → 3 → 5 → 8 → 16 → 32 (16K tokens), random segment counts for length generalization; dropping the
  associative matrix (PRMT) lost the gain. Our T=1024 (2 chunks) is the first step of that curriculum; p25d (4 chunks)
  the next; idea H (a late long-T phase) is the rest of it.
- **Block-Recurrent / Block-State Transformer** (2022/2023) and **Llama-4 chunked attention**: attention only within
  aligned blocks; a recurrent state or a global mixer carries information across blocks.
- **Infini-attention reproduction failed** (HF blog, 2024): linear (non-delta) compressive memory inside each attention
  layer lost cross-segment facts; the balance gate barely moved (needed 30× LR, no weight decay). Our GDN-2 delta memory
  is a stronger store, but the lesson stands: measure the cross-segment job directly.
- **Titans MAC** (2501.00663): memory read-outs prepended as tokens to each segment's attention.
- **HAM** (arXiv 2603.22325): a GDN takes every token and only "surprising" tokens go to the KV cache — a v5 idea for
  dividing labour by surprise.

## 4. Ideas (status)

| # | idea | status |
|---|---|---|
| A | **Chunked attention + full-gradient memory**: attention only inside aligned C-token chunks (`attn_chunk`), memory over the whole T = 2–4 C sequence. Aligned boundaries block relay at every depth (fixes cause 2); full gradient trains the write path (cause 3); every token keeps exactly its T=512 attention context, so the expected ppl cost is ~0 | **BUILT, queued** (r25a/b 150M, p25a–f 6L) |
| B | **Cross-chunk gap MQAR** (`mqar_xchunk_frac`): kv in chunk A, query in chunk B | **BUILT, queued** |
| C | **NL fact episodes** (`nlfact_every`, `src/nl_fact_aux.py`): 7 relations, 89 names, held out from the eval's names and template; 70% cross-chunk | **BUILT, queued** (p25f) |
| D | **2nd memory layer at depth** (contextual keys, cause 4) | queued (p25e) |
| E | **Multi-reader H0**: later layers query H0's state with contextual queries (YOCO-like, r17a idea) | to-build if D helps NL facts |
| F | Titans-MAC memory tokens at each chunk start | idea (v4.x) |
| G | Surprise-gated writes (β from prediction error; HAM/Titans) | idea |
| H | **Long-chunk final phase** for v4 (C 1024–2048 for the last ~10%, memory beyond) | to-test in the v4 config |
| I | **Demo: chunked streaming** — attention over the current chunk, memory over the whole chat; brain view lights the write at the fact and the read at the question; memory-off makes it forget | to-build (needs chunk-aware `step_kv`, now guarded) |

## 5. Queue (launched 10-04 23:25, Glenn's standing OK to queue)

- **5090** (`temp/queue_5090_r25.sh`): r25 evals on r24a / r24b / r21b as each finishes (forced 512 chunks), then
  **r25a** = r21b + chunk512/T1024 + xchunk 0.5 (control r21b) and **r25b** = r21i + the same (control r21i).
- **5080** (`temp/queue_5080_r25.sh`, stops by 17:35 10-05): r23b → p25a (chunk512 + xchunk) → **p25f** (+ NL facts)
  → p25b (chunk only) → p25e (+ H3) → p25d (T2048, 3 boundaries) → p25c (chunk256). Control r23b.
- Every run gets `temp/run_r25_evals.sh`: per-chunk ppl with memory on/off, cross-chunk haystack, NL facts in/beyond
  the window.

## 6. Decision rules (for the v4 config freeze ~Oct 8)

- **Works** = chunk-1 ppl within ~2% of the control, memory Δ on chunks 2+ clearly > 0 (and largest on the first 64
  tokens), cross-chunk haystack well above chance, β not collapsed. Then v4 trains at T=1024 (2 chunks, B32 on the
  H200, same tokens/step) with xchunk MQAR, and the demo uses chunked streaming.
- **NL facts** beyond the window above chance (p25f) → add the NL-fact aux to v4; if only H3 (p25e) gets there →
  memory at depth or multi-reader (E).
- **Fails** at 12L but works at 6L → memory at depth (D/E) at 12L next.
- **Fails everywhere** → fallback: v4 ships with attention recall (r21b recipe); memory stays a streaming bonus.
