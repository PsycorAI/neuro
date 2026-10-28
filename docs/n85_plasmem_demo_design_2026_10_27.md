# Plasmem demo — design (Neuro 85, 2026-10-27)

Goal: one page that makes a developer or investor *see* that Plasmem is a memory made of synapses, not a vector database:
it recalls the right moment, knows when, knows who, says "I don't know", and **gets better the moment you correct it**.
Code: `scripts/memdemo/` (new; `scripts/demo/` untouched). Style: the v4 demo's dark palette and panels.

## What it shows (the five beats)

| # | Beat | What the viewer sees | Backed by (measured) |
|---|------|----------------------|----------------------|
| 1 | **Recall** | Ask a question → the memories that light up, ranked, with the cue words that fired | LoCoMo r@10 76.7 vs keyword + vector 64.6 |
| 2 | **When** | "last week", "in May" → a time-cell strip lights the matching days | time cells, `when()` parser |
| 3 | **Who** | "What did Melanie say about…" → only her memories stay bright | who-cells |
| 4 | **I don't know** | An unanswerable question → confidence gauge drops, "nothing I remember" | abstention AUROC 60.3 vs vector 53.7 |
| 5 | **Learns from use** | A paraphrase misses → click the right memory → the same and *new* paraphrases now hit | F5: 57 → 89 (correct), never-asked 77 → 81 |

## Layout (one screen, two columns, phone = one column)

- **Header:** "Plasmem — memory with plasticity" · chips: memories stored, synapses, file size, front end (see Honesty).
- **Left — Diary.** The stored memories (turns of a conversation), newest last, each with date + speaker. An "add a memory"
  box (text, date, who). Recalled memories glow in the diary with their rank.
- **Left, below — Ask.** Query box, "now" date control, Ask button. Four one-click **tour** buttons (Recall · When · Who ·
  I don't know · Learns) that run scripted scenarios with known answers.
- **Right — Synapse view (the hero).** Three columns of dots joined by lines:
  cue neurons (words + semantic neurons that fired) → memory neurons (diary entries) ← time cells / who cells.
  Line brightness = synaptic drive. Animated in ~1 s: cues fire, drive flows, winners light up.
  Counter under it: "214 of 65,536 semantic neurons fired · 7 word neurons · 0.3 % of synapses used".
- **Right — Answer.** Top-5 recalled memories, each with ✓ "that's it" and "not this one" buttons; a "the right one
  is…" mode lets you click any diary entry. Confidence gauge (abstains below a threshold).
- **Right — Ordinary vector search (comparison).** Same question, same embedder, plain cosine top-5. Green/red ticks when the
  tour knows the answer. Toggle to hide it.
- **Right — Learning log.** After a correction: "question rank 7 → 1; paraphrase 'when did…' rank 12 → 2". Shows the new
  use synapses (count) that were grown.

## Backend

- `scripts/memdemo/server.py` — stdlib `ThreadingHTTPServer` (same pattern as the v4 demo), port **8766**, CPU only
  (`CUDA_VISIBLE_DEVICES=`), so it never touches the GPUs or the v4 demo.
- `scripts/memdemo/engine.py` — wraps `psymem.core.Psymem` + `bench.Embedder` (bge-small, CPU). Adds `explain()`:
  per query returns cue neurons fired (word ids + semantic neuron count), per-memory drive split by channel (word,
  semantic, time, who, neighbour, use), confidence, top-K. Vector-search baseline computed alongside.
- Endpoints: `GET /` page · `GET /api/diary` · `POST /api/remember` · `POST /api/ask` · `POST /api/feedback`
  (confirm / correct → `reinforce`) · `POST /api/reset` · `GET /api/tour`.
- **The memory file:** the diary + synapses saved to `logs/memdemo/<name>.plasmem` (one npz + jsonl) after every change;
  reload on start. This is the "your memory is a file you own" story.
- Data sets: (a) one LoCoMo conversation (~400 turns, real benchmark data with gold answers for the tour); (b) a short
  original diary I write (~60 entries, ours to show publicly); (c) empty — type your own.
- Config: the LoCoMo-winning shared setting (big code + nb 0.5 + time cells). LongMemEval best uses nb 0 — not in the demo.

## Honesty (shown on the page, recorded in disclosures.html)

- Front end chip: **"front end: bge-small (transformer) — spiking encoder in training"** until the spiking encoder
  (s1b / stage 2) is swapped in; then "front end: spiking". Everything after the front end is synaptic.
- No joules claimed: we have no meter for this path yet. Show counts (neurons fired, synapses used, bytes) only.
- Benchmark numbers on the page are labelled with their benchmark and k; tour scenarios are chosen examples, not a score.
- LoCoMo data is CC BY-NC (non-commercial) — fine for a private demo; the public version uses our original diary.

## Out of scope for v1 (ask Glenn)

- A generated *answer* (LLM reader over the recalled memories) — needs the GPU (Llama-3B) which is busy; v1 is recall only.
- Energy meter, multi-user, MCP server.

## Test plan

1. `engine.py` unit check: the tour questions rank their gold memory in the top 5; abstain question below threshold;
   correction moves the gold to rank 1 and a held-out paraphrase improves. → verify: script prints pass/fail.
2. Server smoke: every endpoint via curl; restart reloads the .plasmem file identically.
3. Browser: page at desktop + phone width, both tours run end to end, no console errors.

## Build notes (v1 built 2026-10-27, after Glenn's approval)

- Built in `scripts/memdemo/` (server.py, engine.py, index.html, diary.json, run.sh). Start: `bash scripts/memdemo/run.sh`.
- Data: only our own invented diary (61 turns, 12 chats, Aug 29 to Oct 26). No LoCoMo data in the demo.
- Changed from the mockup after measuring on that diary (temp/n85_memdemo_probe*.py):
  - **Who** is a toggle, not a tour step. On this diary, who cells rarely change a ranking.
  - New tour step **Beats vector search** ("hallway": Plasmem rank 1, vector rank 10).
  - **No abstain verdict.** Confidence does not separate answerable from unanswerable questions here. The gauge is labelled
    experimental; "I don't know" shows the never-stored word (no neuron for "cat").
- Verified:
  - All endpoints work.
  - A restart reloads the .plasmem file with its learned synapses (the learned question stays at rank 1).
  - Reset restores rank 6.
  - Phone width: no horizontal scroll.
  - No console errors.
