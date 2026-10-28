# v4f post-training plan (Neuro 73, 2026-10-22, Glenn: "as fair and unassailable as possible")

Goal: post-train v4f (standard 1B transformer) with exactly the pipeline that made v4's demo model, so v4 vs v4f can be
compared after post-training too (tools, refusals, self-knowledge, chat, brains, ppl, memory). Starts when v4f's KD
phase ends (~10-30 evening). Runs on the 5090 (SFT) and the 5080 (probe, brains, evals). ETA ~2 days -> ~11-01/02.

## v4's pipeline (what is copied)
| step | v4 (what was run) | v4f (same) |
|---|---|---|
| 1. base | LAWA average of 5 conversion ckpts over ~19k steps (`temp/n62_avg_list_cF`: final + 4 earlier, same spacing) vs the final alone; lower chunk-1 ppl wins (`temp/n62_final.sh`) | same steps relative to the end of v4std_conv (93k-step KD phase, same length as v4's) |
| 2. self-knowledge probe | `temp/n62_probe.sh` on the base -> `selfknow_<base>.jsonl` (labels = what THIS model knows) | same procedure on the v4f base (labels differ by design) |
| 3. SFT | `temp/n62_sft2.sh sft_v4_final`: recipe `temp/n62_recipe.env` (text replay 0.15, 60M tok) + `sft_xchunk_frac 0.2`, `sft_xchunk_batch 1`, `muon_lr 1e-4`, `lr 1e-5`; same data blend (tools proto 2, coldstart, selfknow) | same config, data, tokens, seed |
| 4. slim + demo knobs | `sft_v4_final_slim`; demo search bias +1 | same |
| 5. chat brain | LoRA r32 on `sft_v4_final_slim` (`temp/n62_brain.sh`, teacher-written chats), merged -> `sft_v4_chat` | same data / recipe, merged -> `sft_v4f_chat` |
| 6. skill brains | Python (python_sk_F, 13.3M) + health (health_sk_F, 24M) on the chat base | same |
| 7. evals | tool_eval, selfknow_eval, chat quiz, brain quizzes, chunk ppl, recall / haystack, lm_eval, energy-v2 | same scripts, same seeds |

## Fairness points
- **LR check.** v4's SFT LR (10x below normal) was picked by a gate on v4 (`temp/n63_x1_gate.sh`: X1 normal LR vs X2 low LR, 30M,
  Glenn's rule = lowest raw-text ppl that keeps tools / refusals / memory). Pretraining gave v4f its own LR check (10-19), so
  v4f gets the same 2-arm gate and the same rule; v4's LR is kept unless the other arm wins clearly. Disclose.
- Home-field advantage: the SFT recipe was tuned on v4 -> disclose; the gate above limits it.
- Same everything else: data files, token budgets, seeds, LoRA settings, eval scripts and seeds, GPU for each eval.
- Cross-chunk recall SFT batches (`sft_xchunk`) were added for v4's chunked attention; kept for v4f (same data), noted.

## To build before ~10-30 (5080 free 10-23 -> 10-26)
- `temp/n74_v4f_post.sh`: steps 1-7 chained, parameterised by run names, re-runnable, alerts; dry run on the newest
  v4std_dense checkpoint (few SFT steps + one brain step + one eval each) to catch config gaps (e.g. spike keys absent).
