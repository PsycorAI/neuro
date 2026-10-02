# Neuro vs Transformer — comparison scorecard (2026-09-28)

Status legend: **WIN** measured win · **LOSE** measured loss · **TIE** · **UNIQUE** capability transformers lack natively · **CLAIM** plausible but unmeasured · **?** untested.
Evidence is honest (post-leak-fix) 30M screens unless noted. Every row needs a matched transformer: same params, tokens, data, and settings.

## A. Core quality — the gate everything else depends on (importance: highest)
| # | Feature | What to measure | Status now |
|---|---|---|---|
| 1 | Language-model quality at matched params and tokens ("smarter per parameter") | val_ppl, 2+ seeds | **LOSE**. Best 2-layer 1.07×; 3-layer 1.17×; pure spiking 1.45×. |
| 2 | Downstream accuracy | HellaSwag / ARC / MMLU / PIQA at ≥400M | **?** v3 sat at chance, but it was leak-trained. |
| 3 | Scaling trend | Does the gap shrink or grow from 30M → 100M → 400M → 1B? | **?** The gap grew from 2 to 3 layers. |
| 4 | Sample efficiency | Quality and recall per training token, early in training | **Mixed.** We learn recall within 100M tokens; the 2-layer transformer can't, but the 3-layer one can. |

## B. Memory and in-context learning (importance: high — the core story)
| # | Feature | What to measure | Status now |
|---|---|---|---|
| 5 | Immediate associative recall vs N | MQAR N4–64, question right after the facts | **TIE.** 100% for us; the 3-layer transformer is also 100%. |
| 6 | Recall under interference ("remembers conversation details") | Haystack recall: facts, then 64–440 tokens of other text, then the question | **LOSE** with the best-ppl config (chance). **TIE** with plain lif_exact. Fix being tested: gap curriculum. |
| 7 | Long context and length generalization | Train at 512, evaluate ppl and recall at 2k / 4k / 8k | **CLAIM** (constant-size memory). Untested. |
| 8 | Context-boundary isolation | Loss and recall right after a topic switch: does the old topic bleed in? | **?** |
| 9 | Persistent, portable state ("brains") and learning at inference time (Hebbian plasticity) | Save the state after reading a document, reload it later: gain vs no brain, per domain, per exposure length | **UNIQUE** capability. Not yet rigorously measured. |
| 10 | Self-review and improvement of its own answers | Accuracy of draft vs critique vs revise, same protocol for both models | **?** Validated at 350M in the leak era; never compared against a transformer. |

## C. Efficiency and cost (importance: high — decides "cheaper")
| # | Feature | What to measure | Status now |
|---|---|---|---|
| 11 | Training speed and cost per token | Tokens/sec and $ per 1B tokens, same GPU | **LOSE.** 30M: ~12k vs ~99k tok/s. v3 426M: 9.7k vs the transformer's ~30k. |
| 12 | Inference speed | Time to first token, tokens/sec, and memory vs context length | **CLAIM** of a win at long context. Not measurable yet: exact token-by-token decode isn't built (the spiking membrane resets each call). |
| 13 | Energy per token (train / infer) | GPU joules measured; neuromorphic energy via op counts | **CLAIM.** The "85% less" is an op-count estimate valid only on sparse or neuromorphic hardware. On GPUs energy tracks wall-clock time, which we currently LOSE. Integer spikes need a recount. |
| 14 | Activation sparsity | Fraction of neurons active per token | **WIN.** 5–15% active vs ~100% dense. This is a fact, not yet a benefit. |

## D. Alignment and safety (importance: HIGH — Glenn 2026-09-28: the leading AI topic now, after the recent AI escape / sandbox incidents; "alignment" and "interpretability" are to be headline terms once proven)
| # | Feature | What to measure | Status now |
|---|---|---|---|
| 19 | **Interpretability** | Are single spiking neurons monosemantic (one concept each)? Compare against transformer MLP neurons using the same auto-interp protocol. Can the model be steered by clamping spikes? | **CLAIM.** Sparse binary / integer codes are a plausible edge. |
| 20 | **Alignment-training response** | Measured gain from "why" lessons (ORPO + why): refusals, value consistency, generalization to unseen cases. Same data and recipe for both models. | **?** Built. |
| 24 | **Memory auditability and control** | Inspect, attribute, edit and erase what the model learned at inference time (brains). Guaranteed forgetting on reset. No weight changes at inference. | **UNIQUE** candidate. **?** |
| 15 | Honesty: hallucination and calibrated abstention | Abstention eval, factual QA, confidence calibration (ECE) | **?** Eval built. |
| 18 | Jailbreak resistance, incl. multi-turn and state poisoning | Attack success rate. Persistent state could help or hurt, so both directions must be tested. | **?** |
| 25 | Context-boundary / instruction isolation | Injected instructions in documents or tool output don't carry over past a state reset or boundary (prompt-injection resistance). Links to #8. | **?** |

## E. Reliability (importance: HIGH)
| # | Feature | What to measure | Status now |
|---|---|---|---|
| 16 | Repetition and looping | Repetition rate in long generations | **?** |
| 17 | Robustness to noisy inputs (typos, junk tokens) | ppl and accuracy vs token-corruption rate | **?** |

## F. Adaptability (importance: medium)
| # | Feature | What to measure | Status now |
|---|---|---|---|
| 21 | Post-training stability | Recall and quality kept through SFT and preference tuning | **LOSE.** SFT erodes recall; partly fixed with rehearsal. |
| 22 | Tool use and structured output | tool_eval | **?** Built. |
| 23 | Continual learning without forgetting (via brains instead of weight updates) | Add a domain via a brain vs via fine-tuning; forgetting of old skills | **?** Potentially UNIQUE. |

## Positioning (not a metric)
The unique combination is spiking neurons + fast-weight (Hebbian / delta-rule) memory + persistent brains + hybrid attention. That's the narrative; rows 1–23 are the evidence.

## Priorities
1. **Close the losses the pitch depends on:**
   - #1 quality (in progress)
   - #6 recall under interference (in progress)
   - #11 training speed
   - #12 exact incremental decode (unlocks #12 and #13)
2. **Measure the claimed wins:**
   - #7 long context (strongest structural argument)
   - #9 brains
   - #13 energy (honest op counts + GPU joules)
3. **Alignment and safety — prove decisively (HIGH):**
   - #19 interpretability and #20 alignment first, then #24, #15, #18, #25.
   - Every claim needs a matched transformer, the same protocol, 2+ seeds, and ideally a pre-registered eval.
   - Marketing uses "alignment" / "interpretability" only after measured wins.
4. **Reliability, then adaptability:** #16, #17, #23.
