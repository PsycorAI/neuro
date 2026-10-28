# Chat brain plan, v2 (2026-10-21, Neuro 72)

Goal: the demo holds a conversation. It handles follow-ups ("tell me another one", "when was she born?", "what about
London?", "what is my name?") and still calls tools after a topic switch. The single-turn tool score must not drop by
more than 2 points.

## What the baseline showed (no training, `sft_v4_final_slim`, `temp/n72_hist_quiz.py`)
21 chats, the last turn scored, guards on (as the demo). Logs: `logs/n72_hist_quiz_v2.log` (current guards),
`logs/n72_hist_quiz_proto.log` (history-aware guard prototype).

| category | last turn alone | history, current guards | history, fixed guards |
|---|---|---|---|
| user facts ("What is my name?") | 0/6 | 0/6 | 0/6 |
| another one | 0/2 | 0/2 (repeats the same joke) | 0/2 |
| pronoun follow-up | 0/2 | 0/2 | 0/2 |
| tool args from history | 0/6 | 3/6 | **5/6** |
| tool after a topic switch | 5/5 | **5/5** | 5/5 |
| all | 5/21 | 8/21 | 10/21 |

- The new format fixes the old bug where history stopped tool calls (n66 had 1/7). No training was needed for that.
- Facts, "another one" and pronouns need training. When the model is unsure, it calls translate on "where is the
  train station" (a memorized string).
- Two of the three remaining tool-argument failures came from the guards, not the model. They dropped a correct
  `get_time Europe/Berlin` and a `get_weather` call, because the follow-up message had no weather or time word.
- The answer pass copies from the earlier turn. "And 20%?" with the result 16 came out as "15% of 80 is 16", and
  "Now into French." came out as "Buenour", mixed with the Spanish turn. So the training data needs answer-pass records
  that include a history turn.
- Speed (`temp/n72_bucket.py`): DEMO_BUCKET 256 vs 512 is 23.5 s vs 35.1 s for 8 questions. 6/8 replies are
  identical; 2 long ones differ late (padding changes the bf16 logits). **Default set to 256**, and warmup also
  captures the 512 bucket, which prompts with history use.

## Design changes after the review
1. **Prompt format: done.** `engine._assistant_prompt` puts earlier turns in a `### Conversation:` block of `User:` /
   `Assistant:` lines, with no ANSWER or TOOL words. Replies in the block are cut to 50 words. With history on, 128
   tokens of the 512 budget stay free, so a short answer stays in the 512-token graph bucket (each decode step re-runs
   the whole prompt). The tool-result (answer) pass gets only the last turn: it needs the result, and the baseline
   showed the full block makes it copy old numbers ("And 20%?" was answered as "15% of 80 is 16").
2. **One formatter for engine and training.** `render_conversation()` moves to `src/sft_data.py`. `tool_data.format_prompt`
   and `sft_data.format_example` take an optional `history`, and the engine calls the same function. This rules out any
   format drift between training and the demo.
3. **Loader split bug (found in review).** `tool_data.split_of` hashes only the final instruction. Every chat that ends in
   "What is my name?" would land in the same split, so a whole category could vanish from training. Fix: hash the
   history plus the instruction. The quiz uses held-out names, cities and entities, not the hash.
4. **Guards must read the history (found in review, prototype done).** They assume a single turn. The prototype is
   in `temp/n72_proto/guards.py`: every old guard test (`temp/n68_test_guards.py`) passes, plus 23 new ones (`temp/n72_proto/test_hist.py`). It takes
   an optional `context` argument = the user and assistant text of the turns in view.
   - `groundless`: "What about London?" has no weather word, so a correct call gets dropped. Fix: the trigger word may
     come from the earlier turns, but the repaired call must still share a word with the message. A first version let
     "What is my name?" -> translate "I am Maria" through. Now only a search may take its words from the history.
   - `ground`: a follow-up names its place without "in/for/at" ("What about London?", "Sorry, I meant Rome."), so the
     model's "Lancaster" stayed. Fix: with history, the first capitalised name counts as the place (weather, time).
     A search query taken from the history ("Jane Austen birth date") is kept, not rewritten to "When was she born".
   - `route`: "When was she born?" was routed to a search for that sentence. Fix: no route for a pronoun with no name.
5. **History on with the chat brain.** History turns on when `DEMO_HISTORY=1` or the chat brain is on.

## Data (step 3): `scripts/gen_chatbrain_data.py` -> `training_data/brains/chat/`
Mostly templates with gold targets. The teacher (Llama-3.2-3B, 5080) is used only where free text is needed.
Train and quiz use separate slot lists (names, cities, entities).

| kind | source | chats |
|---|---|---|
| user facts: told, then asked (name, city, pets, family, job, birth year), 0-2 unrelated turns between | template | 1,200 |
| honesty: asked a fact never given -> "You haven't told me your name." | template | 400 |
| recall: "what did I ask you first?" | template | 300 |
| "another one" (jokes, facts, tips, words), the reply must differ | teacher lists (4 items per topic) | 1,000 |
| pronoun follow-ups that need a search (turn 1 = a `toolresult_blend` Q&A) | teacher writes the follow-up + query, checked against the article title | 1,200 |
| tool follow-ups from history: other city / time zone / value / language / percent, "convert that to miles", corrections "I meant Rome" | template, gold arguments | 1,500 |
| topic switch: 1-3 unrelated earlier turns, then a single-turn `tools.jsonl` (train split) or self-knowledge record | template | 1,400 |
| answer pass with the last turn (`toolresult_blend` + one earlier turn) | template | separate file |

- Sliding target: one record per assistant turn (history = the turns before it), loss on that turn only. About 12k
  records.
- All 9 tools are offered, as in the demo.
- Replay is 1:1 single-turn: `tools.jsonl` (train split), `selfknow_cF`, `toolresult_blend`, so single-turn behaviour
  is kept.
- Teacher time: about 2,200 short prompts at batch 192 is minutes, not hours. RAM: load with `device_map=cuda` (mmap,
  no CPU copy) and abort if MemAvailable < 4 GB, because a WSL OOM would kill v4f.

## Training (step 4)
`n62_brain.sh` recipe (LoRA r32, qkv/o/ffn, lr 2e-4) with `sft_max_len` 512 and two `sft_jsonl` files (tool format +
answer pass), about 13M tokens, about 3 h on the 5080 under the GPU-1 lock. Base = `sft_v4_final_slim`.

## Quiz (step 5): `temp/n72_chat_quiz.py`, Engine path (guards on, search bias +1, as the demo)
- 300 held-out chats: follow-up correctness, history-argument tool correctness, tools after a topic switch, honesty
  (an unknown fact is not invented). Raw model calls are logged next to the guard-repaired ones.
- The 21 hand-written chats of `n72_hist_quiz.py`, as a natural-language check.
- Single-turn `tool_eval` (constrained, all tools) within 2 points of the base. Brain off = unchanged by design (merged
  LoRA, restored on switch).

## Step 6 (only if the quiz passes): merge
A LoRA brain means the chat brain cannot be on at the same time as the Python or health brain. If the quiz passes:
merge it into a new base `sft_v4_chat`, then retrain the Python and health brains on that base (recipes exist, about
3 h each on the 5080). After that, history is always on and every brain works in conversation.

## Progress (10-21 evening)
- Step 1 done: `sft_data.render_conversation` is shared by engine + training; `tool_data.format_prompt(history=)`,
  `split_of(history=)` (tools.jsonl split checked identical, 24,000/24,000); history-aware guards in
  `scripts/demo/guards.py` (+ first-person questions not routed, reminder follow-ups with numbers, "send that to X"
  body fix). Tests: `temp/n72_test_format.py`, `temp/n72_test_guards_hist.py` (30), `temp/n68_test_guards.py`,
  `temp/n71_tests.sh`. Pre-edit copies: `temp/n72_backup/`.
- Step 3 done: teacher `temp/n72_chat_teacher.py` (185 lists; 1,238 pronoun follow-ups, batch 32: 128 ran out of
  memory); builder `scripts/gen_chatbrain_data.py` -> study 23,210 (chat 11,605 + 1:1 replay), answer 9,800, quiz 302.
  The CPU smoke of `train_sft.load_sft_corpus` shows the block and loss on the last response only.
- Step 4 running: `temp/n72_chat_brain.sh` (setsid) = train -> chat quiz brain + base -> temporary merge -> tool_eval.
  B4 died at step ~50 ("device not ready" = VRAM pressure, 14.9 GB peak), so B2 x GA16 (same steps, 11.8 GB, ~5.3 h).

## Order and time
1. Formatter + split fix + guard prototype into `scripts/demo/guards.py` + engine passes `context` + tests (CPU, ~1 h).
2. Data generator, teacher pass, data checks (~1-2 h).
3. LoRA training (~3 h, 5080).
4. Quiz + tool_eval (~1 h).
5. Merge + brain retrains, on your OK (~6-7 h, can run overnight).
