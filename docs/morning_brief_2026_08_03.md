# Morning brief — 2026-08-03

## What happened overnight

**Phase 0 (compile parity gate) PASSED.** val_ppl 142.10 vs known eager
140.30 (+1.28%, inside the 2% gate); memory-Δ +252 vs +217 (better);
spike identical. torch.compile is quality-neutral with the attention block.

**Phase 1 (attention-ratio A/B) DECIDED — 7:1 (1 attention layer) wins.**
val_ppl was tied (~48), but recall showed a massive gap:

| Arm | Recall N=4/8/16/32/64 (--replay split, B=64) |
|---|---|
| A (7:1) s0 | 100/95/78/75/59% |
| A (7:1) s1 | 100/98/94/80/47% |
| **B (3:1) s0** | **31/22/9/6/3%** (~chance) |
| **B (3:1) s1** | **52/22/11/5/3%** (~chance) |

Confirmed via without-split-replay control (arm A: 97/89/28% vs arm B:
41/8/2% at N=4/16/64). The 3:1 arm just failed to learn associative recall.
Details: memory `neuro_phase1_attnratio_finding.md`; §"Phase 1 result" in
`docs/step3_midscale_design.md`.

**Corollary: memory-Δ ALONE is not a memory-pathway health check.** Arm B
had HIGHER memory-Δ (+255 vs +231) while having ~zero recall. Always pair
with recall_eval.

**λ=0.5 CONFIRMED at old shape (90M, 2 seeds).** val_ppl tied (55.9 vs 55.3),
spike 0.204 vs 0.238 — closer to target 0.15.

**λ=0.5 VERIFY AT NEW SHAPE (d=640/n=1536/L=8) — seed 0 done, wins on ALL
metrics AND on recall.** The early startup memory-Δ dip resolved completely:

| | λ=0.3 (a1_s0) | λ=0.5 (new-shape s0) |
|---|---|---|
| val_ppl | 48.56 | **44.78** (−8%) |
| memory-Δ | +231 | +236 (+2%) |
| spike | 0.278 | **0.160** (at target 0.15) |
| Recall N=4/8/16/32/64 | 100/95/78/75/59% | 100/100/92/72/**69%** |

λ=0.5 recall matches or beats λ=0.3 at every N, and is decisively better at
the hardest N=64 (69% vs 47-59%). **Full confirm at the new shape.** Seed 1
still running on the 5080; expected to track s0 given how tight the
existing 2-seed data has been.

The memory-Δ tripwire I proposed for Step 3 is no longer necessary — the
recall check settles it.

**276M throughput probe at block=640 done.** Block 640 hits WDDM ceiling
(batch 24 = 30 GB, batch 28 = spill). Block 512 at batch 28 = 17,690 tok/s,
27 GB — 8% faster AND safer margin. **Keeping block=512 for Step 3**; block
640 deferred to Step 4 / 1B planning.

## Current state

- **5090**: running `screen_attnratio_90m_a2b_s0` — placement-ablation
  follow-up (arm B with attention at `[2, 5]` instead of `[3, 6]`, single
  seed, 150M tokens, ~3h). Scoping test for Step 4 / 1B design ("was it
  position or count that broke recall?"), does NOT affect Step 3.
- **5080**: running `screen_lambda05_newshape_s1` (seed 1 of new-shape λ
  verify — seed 0 done, wins on everything including recall)
- Phase 1: all 4 runs complete, checkpoints saved
- Phase 2 config: `configs/step3_267m_s0.yaml` written, verified builds
  correctly (276.5M params, arm A, λ=0.5, block 512), NOT launched

## Recommended plan (needs your sign-off)

Simplified from the earlier version since λ=0.5 is now confirmed on recall
at the new shape:

1. **Let the 5080 λ verify seed 1 finish** (~6h from now) — 2-seed
   confirmation of the s0 result. Should be a formality; s0's data is
   strong enough that seed 1 mostly checks for outliers.
2. **Launch Step 3 seed 0** with `configs/step3_267m_s0.yaml` on the 5090
   as soon as its current placement-ablation completes (in ~3h). No
   tripwire needed — recall data settles the λ=0.5 choice cleanly.
3. Chain Step 3 seeds 1 and 2 after seed 0 (or launch each as soon as the
   prior one finishes).
4. Run `recall_eval` on λ verify s1 as it completes (placement ablation
   already done, see below).

## Extra results run overnight while GPUs were free

- **Placement ablation ([2,5] vs [3,6] on arm B, 1 seed, 150M tokens)** —
  DID NOT SAVE the arm-B recall disaster. `[2,5]` gave 16/9/6/2/0% recall
  (WORSE than `[3,6]`'s 31/22/9/6/3%) and worse val_ppl too. **Conclusion:
  the problem is the COUNT of attention layers (2), not the position.**
  For 1B / Step 4 planning: don't chase 3:1 without a fundamentally
  different recipe change (larger n_neurons, higher MQAR weight, different
  attention design). Ruling out placement removes a degree of freedom.
- **Recall on λ=0.5 OLD-shape checkpoints** — recall-neutral vs λ=0.3
  baseline (98-100/95/77-86/61-72/34-48% vs 100/86/80/73/53%). Tail
  variance at N=64 is noisier but no regression. Combined with the
  decisive new-shape recall win, λ=0.5 is recall-safe at both shapes.

## Files touched

- `src/hybrid_attn.py` — QK-norm (done earlier)
- `configs/screen_qknorm_30m_s0_compile.yaml` — Phase 0 gate
- `configs/screen_attnratio_90m_a{1,2}_s{0,1}.yaml` — Phase 1 (all 4)
- `configs/screen_lambda05_90m_s{0,1}.yaml` — λ confirm (old shape, done)
- `configs/screen_lambda05_newshape_s{0,1}.yaml` — λ verify (new shape, running)
- `configs/step3_267m_s0.yaml` — Phase 2 (ready, not launched)
- `docs/step3_midscale_design.md` — full design + all results
- `docs/morning_brief_2026_08_03.md` — this file
- Memory: `neuro_phase1_attnratio_finding.md`, `neuro_step3_midscale_plan.md`

## Concerns worth calling out

- **Attention-count question is not fully answered for the 1B design.** The
  1B is still "one attention layer at the middle of the stack"; that's fine
  for coherence-limited use, but if a future goal calls for long-context /
  in-context recall (e.g. RAG), the current recipe won't compete with 3:1
  hybrid families. A placement-ablation follow-up (2 attention layers at
  even spacing rather than clumped) would answer whether more attention can
  EVER work here.
- **λ=0.5 memory-Δ dip at new shape** — first eval suggested a real drop
  (269 vs 415). Full trajectory pending. If it settles at a low absolute
  value (<150), λ=0.3 might be safer for Step 3 despite the confirm at old
  shape.
