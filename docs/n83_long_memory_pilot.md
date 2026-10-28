# n83 long-memory SFT pilot (2026-10-27)

Goal (Glenn 10-26): teach v4's spiking memory layer to carry facts stated in plain language past the 1,024-token
attention window, trained on 4,096-token sessions; v4f gets the same data at the same length later (fair match).

## Why (n82 diagnosis)
Memory ablated = identical recall at every distance: the memory adds nothing for plain-language facts. Its write gate
barely separates facts (0.27) from other text (0.23); it opens on the synthetic MQAR key/value tokens it was trained on
(0.39). Raising the write floor or widening attention at inference does not help.

## Data (temp/n83_lm_data.py -> data/n83_longmem/{train,val}.npz)
- 4,096-token episodes in the n82 long-session format (`### Instruction: ... ### Response: ...` turns).
- 2-8 facts per episode ("Please remember this: my X is V." and 7 other phrasings; varied acks), placed mostly in the
  first half. Every fact is asked about (30% twice): chat form ("What is my X?" -> "Your X is V.") or the n82 cue form.
- 75% of questions >= 1,100 tokens after the fact (attention is chunked at 1,024 in training, so memory is the only
  path), 25% within 30-900 tokens (attention's job, kept).
- Filler: alpaca-cleaned turns (70%) and search_pack passages + summary (30%); 10% of episodes plain pretraining text.
- Curriculum: first 30% of episodes keep questions within 2,048 tokens of the fact (RMT practice: short first).
- HELD OUT: 10 relations (the n82 five + five) and their values never appear in training.

## What trains (temp/n83_lm_train.py)
- Arm `mem`: only the memory block (block 6, ~?M params); everything else frozen. Safest for perplexity.
- Arm `memup` (if `mem` stalls): + the blocks after it at 0.1x LR (the read side).
- Canonical NorMuon (matrices, muon_lr 1e-4) + AdamW (gates/norms, lr 1e-4: the low-rank write/erase gates live here).
- Loss = mean CE on answer value tokens + 0.25 x weighted CE on replies/text; + 15% of micro-steps one cross-chunk
  MQAR batch at 0.15 (keeps the synthetic memory skill / demo memtest).

## Evals
- temp/n83_eval.py: demo regime (streaming decode, sliding window 1024, bf16), held-out facts, distances
  0..8,000 (8,000 = 2x the training length), cue + chat probes, memory OFF at 1,500 / 3,000, demo memtest.
- val accuracy by distance during training (memory on/off), memory-block spike rate (energy).
- Perplexity: temp/n61_lawa.py base vs trained, same windows.

## Cost
Pilot ~15M tokens on the 5080 (5090 busy with v4f's run). v4f matched run later on the 5090: same data, same length.

## Open
- Demo integration: the trained model would need the int8 + prune steps re-applied and the brains re-checked.
- The demo builds its stream with a history block; the eval uses the n82 plain concatenated format.

## Results
**Baseline sft_v4_chat** (logs/n83_eval_base.log): held-out facts 5/5 (cue) at 0 and 600 tokens, 0/5 from 1,000 on (both
sets), memory OFF 0/5, memtest 19/20.

**Pilot v1 `n83_lm_mem`** (memory block only, 15M tokens, 2.36 h on the 5080, logs/n83_lm_mem.log):
- val (training relations, unseen episodes; whole value right): 1-2k tokens back 7% -> 48%, 2k+ 8% -> 54%, <1k 51% -> 81%.
  Memory OFF at the end 8-10% = the gain is the memory. Plateau from ~step 250 (8M tokens).
- held-out relations, demo regime (logs/n83_eval_n83_lm_mem.log): beyond the window 1/5 per set at most distances
  (vs 0/5), e.g. "Thumper" recalled at 1,500 / 3,000 / 8,000 tokens; many near misses ("Biscuits", "Lisbona", "6167"
  for 6182) and fact mix-ups ("Your sister's name is Biscuit"). Weak transfer: the memory learned the training
  relations, not "any fact". Memory OFF still 0/5.
- perplexity unchanged: full 18.942 -> 18.922, chunk1 19.214 -> 19.055, chunks2+ 18.674 -> 18.790 (logs/n83_ppl.log).
- memtest 19/20 -> 20/20; memory-block spike rate 0.0130 -> 0.0128 (no energy cost).
- Fix made: the trainer saved its own cfg (compile off -> an unused LIF module) and the strict demo loader refused it;
  cfg now = the base's (temp/n83_fix_cfg.py repaired the v1 checkpoint).

**Arm v2 `n83_lm_mem_lr3`** (same, muon/adam lr 3e-4, 15M tokens, 2.4 h; logs/n83_lm_mem_lr3.log): val 1-2k 74%,
2k+ 84%, <1k 93% (no plateau yet). Held-out (logs/n83_eval_n83_lm_mem_lr3.log), cue / chat hits of 5:
- set A (the n82 five): 1,000: 4/4 | 1,500: 4/4 | 2,000: 4/4 | 3,000: 4/3 | 4,000: 4/4 | 6,000: 3/3 | 8,000: 3/2
- set B: 1,000: 4/4 | 1,500: 3/3 | 2,000: 1/3 | 3,000: 1/2 | 4,000+: 0/0
- memory OFF 0/5 at 1,500 and 3,000 (both sets) -> the recall is the memory's. memtest 20/20.
- hits are substring matches: many are "Clara Barton", "Biscuits", "4719"+extra digits; the memory carries the word's
  start more reliably than its exact end. Memory-block spike rate 0.0130 -> 0.0166 (+28%; the memory block is a small
  share of the model's work).

**Arm v3 `n83_lm_v3`**: 127 training relations instead of 35 (N83_DATA_V3=1 -> data/n83_longmem_v3), 3x LR, 12M
tokens (1.86 h), aimed at the held-out transfer. val (its own relations) 1-2k 59%, 2k+ 75%. Held-out NOT better than v2:
set A 1,000: 4/2 | 2,000: 3/3 | 4,000: 3/1 | 8,000: 2/1; set B 1,000: 2/1 | 2,000: 2/2 | 4,000: 1/1 | 8,000: 1/0.
OFF 0/5, memtest 20/20. (Fewer tokens than v2, and 5 facts per set is a small sample.)

**Perplexity** (logs/n83_ppl2.log, same windows): base 18.942 | v2 lr3 18.717 | v3 18.840 (full; chunk1 / chunks2+ also
equal or lower). No perplexity cost.

**Best so far: v2 `n83_lm_mem_lr3`.** Next: more tokens at 3x LR (no plateau yet), a held-out eval with more facts
(5 per set is noisy), v4f's matching run on the 5090 (same data, `--arm full`), then demo integration (int8 + prune
re-applied, brains re-checked).

**TEAL** (temp/n83_teal.py, logs/n83_teal_v4.log): v4's linear-layer inputs are already 77.3% exact zeros (spike codes);
TEAL targets 25/40/50% give val ppl 18.436 -> 18.472 / 18.719 / 19.060 (the few non-spike-coded inputs get thresholded).
v4f (n74d_sft_v4f_chat, logs/n83_teal_v4f.log): 0% natural zeros; ppl 28.365 -> 28.512 / 29.251 / 30.064 (+0.5% /
+3.1% / +6.0%) vs v4 +0.2% / +1.5% / +3.4%. (Load keys not checked in that script; the v4f chat SFT is an early one.)
