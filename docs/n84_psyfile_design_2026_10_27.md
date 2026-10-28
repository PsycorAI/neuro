# Psyfile v2 design: "a brain that keeps a diary" (Neuro 84, 2026-10-27)

Status: PROPOSED (Glenn Q16-Q22, 10-27). Nothing built. Replaces the earlier proposal where memory.safetensors was the core and the transcript optional.

## The problem today
Three separate stores that overlap:
- saved session = text replay
- brain file = 16.6 MB memory state: model-locked, fades after ~2k tokens, carried by HOLA not the Hebbian matrix, merging fails
- LoRA brains = slow to make, lossy, model-locked

Nothing checks that any of them actually remembers.

## Facts the design rests on
- Text is the only exact, model-independent part, and the smallest: ~1.3 bytes/token zipped. A 10k-token session is ~13 KB, against the 16.6 MB state.
- Verbatim beats summaries for recall: 91% vs 14% exact match (arXiv 2601.00821).
- Fine-tuning stores facts poorly unless each fact is seen ~100-1,000 times in varied wording:
  - Physics of LMs 3.3: ~2 bits/param, but only at ~1,000 exposures.
  - Knowledge-injection papers (2508.06178, NAACL-F 2025) agree.
  - RAG beat fine-tuning for injecting facts (Ovadia 2023).
- A recurrent state is a cache of its inputs (ReplaySSM, 2026: "cache inputs, not state").

## The design: three text layers + optional lenses
A Psyfile (.psy, a ZIP) holds:

1. **Diary** (required): the exact words, turn by turn, with timestamps and attachments. Lossless. The master copy.
2. **Notes** (required): written by the brain itself.
   - While talking: per-span salience from its own plasticity (memory write strength beta, surprise, HOLA retention), plus links between spans that were recalled together (Hebbian co-activation).
   - While dreaming: cue questions per span.
3. **Quiz** (required): the recall contract.
   - Grounded question -> answer pairs, each citing a diary span and verified open-book.
   - Each item carries a spaced-repetition strength.
   - Deleted spans become must-fail items.
4. **Lenses** (optional): compiled, model-specific caches, each tagged with the model hash and always rebuildable from 1-3:
   - fast-state cache (today's brain file);
   - personal adapter (LoRA);
   - cue-embedding cache.

## Operations
- **Talk**: notes are written live (learning while talking).
  - Recall = match the question to cue notes, then quote the exact diary span with its date and source ("show your work").
- **Open** (new session or new model): replay the top-tagged spans to rebuild the fast memory.
  - Works on any Neuro version; a transformer simply reads them.
  - Costs a fraction of re-reading everything.
- **Dream** (idle and charging):
  1. Take the quiz closed-book.
  2. Check answers open-book against the diary.
  3. Train the adapter only on failures (with paraphrases), on a spaced schedule.
  4. Pass a do-no-harm gate (general ppl/eval), or roll back.
- **Forget**: delete the span; its quiz items flip to must-fail; rebuild the adapter without it. The quiz then proves it is gone (forgetting receipt).
- **Migrate**: the new model takes the quiz and dreams only what it fails. No weight translation.
- **Merge**: union of diaries, notes and quizzes. Conflicting answers are flagged; the newer one wins unless the user decides otherwise.
- **Other AIs**: use the diary and quiz as retrieval context. Open-weight models can dream too.
- **Skills / brains**: same format.
  - The diary is a book, manual or corpus; the lens is the skill adapter; the quiz certifies the skill.
  - A personal Psyfile lists the skills it uses.

## Dreaming risks -> mitigations
| Risk | Mitigation |
|---|---|
| Wrong recollections get trained in | Train only on quiz answers verified against the diary; the diary stays ground truth; nothing uncited is trained |
| Forgetting general skills | Per-user adapter with the base frozen; failures only (smallest change); rehearsal mix; do-no-harm gate with rollback |
| Forgetting older personal memories | The quiz is cumulative; spaced re-tests catch regressions |
| Privacy | Adapter per user, on device, never merged into the shared base; encrypted diary; redaction; forgetting receipts |
| Battery / cost | Dream while charging; failures-only keeps runs small |
| Facts change over time | Time-stamped items; newer supersede older; contradictions shown to the user |

## Can a LoRA hold a whole session?
- **Capacity**: a rank-16 adapter on v4 (d 2048, FFN 12,288, 12 attention blocks) is ~9M params (~18 MB). The ideal is ~2 bits/param, ~2 MB of facts, against ~40 KB of raw text per 10k-token session. So there is room.
- **In practice** that density needs ~100-1,000 exposures per fact. Verbatim recall is unreliable ("How LoRA Remembers", 2605.30260: needs p > 0.5 per token). Who / when / exact words are lost.
- **Split**: the adapter holds gist, habits, preferences and frequently used facts; the diary holds the full session.

## Sizes (estimates)
| Part | Size |
|---|---|
| Diary | ~13 KB per 10k-token session; ~5 MB per year of daily 10k-token chats |
| Notes + quiz | ~+5-10 KB per session |
| Fast-state lens | 16.6 MB today (int8 / low-rank could halve it); optional, rebuilt in seconds |
| Personal adapter | ~5-20 MB per person (not per session); optional |

## What is new (honest)
**Known** (cite, don't claim):
- verbatim storage;
- question-indexed retrieval (doc2query, hypothetical questions);
- self-study into adapters (Cartridges, GenerativeAdapter ICLR 2025, MemOS plaintext -> parametric);
- memory types and migration (MemOS MemCube);
- an open memory format with signing and verifiable deletion (MacPaw Portable Memory, July 2026 proposal; it has no recall tests);
- cross-model KV transfer (C2C, ICLR 2026);
- replaying inputs to rebuild state for speed (ReplaySSM);
- hippocampal indexing for RAG (HippoRAG);
- associative-memory LMs (MeMo).

**Not found** (searched 10-27; may exist somewhere):
1. notes written by the model's own plasticity signals onto verbatim text;
2. rebuilding fast memory on another model version by replaying only tagged spans;
3. a recall contract inside the memory file that certifies any model, consolidation and forgetting (behavioural forgetting receipt);
4. quiz-gated, failures-only, spaced dreaming;
5. one format for personal memories and skills, with weights as optional rebuildable lenses.

New as a combination; parts 1-4 look new.

## VISION DRIFT (flag)
The master copy becomes text, not synapses. The neural parts stay central: tagging (plasticity), fast-memory rebuild (Hebbian state) and dreaming (slow plasticity). Framing: a brain that keeps a diary.

Risk: notes are only as good as the write gate. The 10-27 brain test showed v4's Hebbian matrix is not load-bearing, so T1 decides this. If v4's gate does not find facts, the v4.5 conversion matters for Psyfiles.

## Tests (proposed; 5080, inference + small LoRA; after the 150M screen; need Glenn's OK)
- **T1: do plasticity notes find the facts?**
  - Log block-6 beta, surprise and HOLA retention over the brain-test sessions.
  - Measure the precision of the top-10% tags on fact spans, against a surprise-only baseline.
- **T2: replay rebuild.**
  - Rebuild from the top 10% / 20% tagged spans.
  - Compare held-out fact recall against the full brain file (3/5) and no memory (0/5); record rebuild time.
- **T3: cross-model.** Replay v4's tags into r26o, n84 c9 and n83, then measure recall.
- **T4: quoted recall.**
  - Fresh session; question -> cue match -> exact span -> answer.
  - Measure recall against distance (expect no fade) and token cost.
- **T5: dreaming.**
  - 50 facts; a quiz-gated failures-only adapter against a naive self-study adapter.
  - Measure quiz score and general ppl change.
  - Then forget 5 facts, rebuild, and check the forgetting receipt.

## Not chosen (parked)
- **A shared "universal memory space"** that makes the neural state itself portable across models: all Neuro models trained against one frozen memory code, or C2C-style projectors. Research-heavy, and the quiz-migrate path gets most of the benefit.
- **A Hebbian pointer index**: values = addresses into the diary. Vision-aligned; test after T4 against a plain vector list.

## Sources
- https://arxiv.org/pdf/2601.00821 (verbatim vs summary)
- https://arxiv.org/pdf/2404.05405 (2 bits/param)
- https://arxiv.org/abs/2508.06178 (knowledge injection)
- https://tridao.me/blog/2026/replayssm/
- https://arxiv.org/pdf/2507.03724 (MemOS)
- https://research.macpaw.com/publications/portable-memory
- https://arxiv.org/html/2510.03215v1 (C2C)
- https://arxiv.org/html/2411.05877v1 (GenerativeAdapter)
- https://arxiv.org/html/2405.14831v3 (HippoRAG)
- https://arxiv.org/pdf/2502.12851 (MeMo)
- https://arxiv.org/pdf/2506.06266 (Cartridges)
- https://papers.cool/arxiv/2605.30260 (How LoRA Remembers)
