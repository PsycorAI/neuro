# Memory product plan (codename Psymem) — 2026-10-27, Neuro 84

**Status: APPROVED to execute** (Glenn 10-27: "focus on the memory product and showcase it with our model").
Brainstorm basis: `docs/n84_psymem_2026_10_27.md`.

## Success criteria (Glenn)

1. **Equal or beat ordinary memory search** on evidence retrieval and end-to-end QA.
   - Ordinary memory search = BM25 + dense embeddings + hybrid (RRF), with the same embedder inside Psymem, so only the
     memory mechanism differs.
2. **Win clearly at two or more** of the differentiating features (list below).
3. **Keep finding and developing other advantages we can advertise** (backlog below; add to it as we go).

## Architecture v0: a two-layer spiking associative memory + diary

| Part | What it is |
|---|---|
| Diary | Exact turns with dates (the master copy) |
| Cue neurons C | Lexical assemblies (one sparse assembly per stemmed word / entity) + semantic assemblies (FlyHash: sparse random expansion of the embedding + k-winners-take-all; the fruit-fly olfactory circuit, Dasgupta et al. 2017) + time cells (session / date) |
| Span neurons S | One per diary turn (localist in v0; fixed-size distributed addresses later) |
| Synapses C→S | Hebbian write at store time; heterosynaptic normalization (busy cue neurons weaken, which is IDF emerging from a synaptic rule) |
| Synapses S↔S | Temporal neighbours (the order turns were stored in), co-recall links (use), dreaming links |
| Recall | Cue → S activation → k-WTA → 2–3 steps of spreading activation S→C→S and S↔S (pattern completion; like HippoRAG PPR but synaptic) → ranked quoted turns + strength |
| Abstain | If peak activation is below threshold → "I don't recall" |
| Learn with use | Confirmed recalls strengthen C→S and link co-recalled spans; disuse decays |
| Dreaming | Replay salient spans; strengthen links that co-activate; prune weak ones; gist entries linked to sources; contradiction flags. Every change is quiz-gated |

- **Writes need no LLM calls:** a write is pure synaptic updates.
- **Energy:** count synaptic events (accumulate-only) vs dense multiply-accumulates.

## Steps (each with its check)

1. **Harness + ordinary-search baselines** (`psymem/bench.py`). LoCoMo (10 conversations, 1,986 Qs) and LongMemEval-S
   (470 non-abstention Qs, ~494 turns each).
   - Check: numbers in `logs/n84_membench.log`.
2. **Psymem v0 core** (`psymem/core.py`, CPU numpy / sparse): lexical + FlyHash assemblies → span neurons, no spreading yet.
   - Check: ≥ hybrid on r@10 overall (criterion 1 gate).
3. **Spreading activation + time cells.**
   - Check: wins on multi-hop (LoCoMo cat 1 all@10; LongMemEval multi-session) and temporal (LoCoMo cat 2; LME
     temporal-reasoning).
4. **Abstention + knowledge update.**
   - Abstention: LME `_abs` + LoCoMo adversarial (cat 5).
   - Knowledge update: newer write depresses older link; LME knowledge-update.
5. **Learn-with-use protocol.** Ask, give feedback, then ask paraphrases; Psymem should rise while the baseline stays flat.
   Plus the forgetting receipt (delete → quiz proves gone).
6. **End-to-end QA:**
   - fixed local reader (Llama-3.2-3B-Instruct or Phi-3.5-mini), same top-k for every system;
   - judge = exact/F1 + an LLM judge (same judge for all).
7. **Scale + cost:** LongMemEval-M / BEAM-1M — size on disk, ms per recall, synaptic events per recall, laptop CPU.
8. **API:** Python SDK + MCP server (remember / recall / associate / dream / forget / explain / quiz). One Psyfile on disk.
9. **Showcase with our model** (separate script; the demo server stays Glenn's):
   - v4 chat alone forgets past its 1,024-token window (n82);
   - v4 + Psymem recalls across sessions with quotes and dates;
   - the energy meter shows the recall cost.
10. **Dogfood:** use Psymem as Claude Code's memory for this project via MCP (our own MEMORY.md / handoff pain is the use case).

**Order of the go/kill decision:** after steps 2–5 (~1–2 weeks of CPU work, almost no GPU).

## Differentiating features (win at two or more)

| # | Feature | Test |
|---|---|---|
| F1 | Multi-step links | LoCoMo cat 1 all@k; LME multi-session sr@k |
| F2 | Remembers when | LoCoMo cat 2; LME temporal-reasoning |
| F3 | Keeps up with changes | LME knowledge-update |
| F4 | Knows what it doesn't know | LME abstention; LoCoMo adversarial |
| F5 | Gets better with use | Feedback / paraphrase protocol |
| F6 | Small, fast, low energy, offline | MB, ms, synaptic events at 1M+ tokens on CPU |
| F7 | Zero-LLM writes | Store cost: tokens / $ / ms per write vs Mem0-style LLM extraction |

## Advantages backlog (develop + advertise; keep adding)

- **Shows its work:** the synapse trace of why each memory came back.
- **Provable forgetting:** receipts (privacy laws).
- **One portable file, any model:** via MCP; migrate by quiz.
- **Dreaming:** finds contradictions and writes linked gists.
- **Shared memory for agent teams:** per-agent synaptic views and permissions.
- **Calibrated confidence:** synaptic strength = how sure.
- **Deterministic and auditable:** same input, same memory.
- **Live "watch your memory grow" synapse viewer:** the viral piece.
- **Neuromorphic-hardware path:** sparse, accumulate-only.

## Go-to-market if it passes (Q33, proposed)

1. **Agent developers through a free open-source MCP server.** Coding agents first (Claude Code, Cursor, Codex):
   - they forget a project between sessions (our own MEMORY.md / handoff pain);
   - install is one line;
   - early adopters spread it on GitHub / HN.
2. **Agent frameworks:** a memory backend for LangGraph, CrewAI, the OpenAI Agents SDK, Mastra. Includes multi-agent shared
   memory with per-agent views.
3. **Game NPC showcase with our own model:** characters that remember you across play sessions, on device.
4. **Then:**
   - enterprise (support agents that remember customers; audit + provable forgetting);
   - neuromorphic / edge chip and device makers (licensing);
   - companions last (regulated: CA SB 243, NY).

**Name candidates (Q32, to-decide; no trademark or domain check yet):** Remi, Myelin, Mindwire, Reverie, Hebbi, Dendra, Mneme.

## Parked (Glenn 10-27: "extraneous work can wait")

- **v4f (v4std_conv) phase B:** paused at ckpt step_0231016 (7.57B of 10.51B tokens, 72%).
  - Finishing it = ~3.2 days of 5090 + KD teacher, then v4f post-training.
  - The queue is frozen (pid 135860).
  - Resume only on Glenn's word: `temp/n84_kd_resume.sh`.
- **150M conversion screen (v4.5):**
  - c9 is trained (`models/checkpoints/n84_c9/final.pt`), but its eval has not run;
  - ctrl / c6 / c12 have not started;
  - resume = `temp/n84_resume_screen.sh`.
- **Psyfile T1–T5 tests:** folded into the steps above.

## Honesty

- Credit FlyHash, Willshaw, HippoRAG, HeLa-Mem, Letta, Larimar and Mem0 benchmarks where used.
- Benchmark claims in this field are mostly vendor self-reports; compare like with like and publish our harness.
- Do not say "neuromorphic" unless the synapses measurably do the recall work.
