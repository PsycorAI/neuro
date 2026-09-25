# Rejection-Sampling Fine-Tuning (RFT) — Design Doc

**Status:** SCOPING ONLY (2026-07-23). No implementation yet — this doc names the key decisions so we can commit to a design before building.

**Goal:** After SimPO finishes the 500M v3 chain, bootstrap further quality by generating N candidate responses per prompt, filtering to the "good" ones via a verifier, and SFT-ing on the filtered set. Widely used in 2026 post-training stacks (see e.g., DPO-Isnt-Enough survey and the Understanding-KD-in-Post-Training paper). Complements SimPO by working on the *distribution of what the model actually produces*, not just relative preferences.

## Loop shape

For each prompt in some dataset P:
  1. Sample N completions from the current model (SimPO'd 500M).
  2. Score each completion with a verifier V(prompt, completion) → real number or accept/reject.
  3. Keep the top-k completions (or accept-only) per prompt.
  4. SFT on {(prompt, kept_completion)} pairs, teacher-forced.
Repeat 1-4 for R iterations if bootstrapping further, or one round for a single pass.

## Key design decisions

### 1. **Verifier V** — the biggest choice, and the hardest

Three options, ranked by cost/quality tradeoff:

**Option A — External LLM as judge (Claude / GPT-4 / Llama-3-70B)**
- **Quality:** Strong. Modern LLM judges correlate ~0.7-0.85 with human preference.
- **Cost:** API $$$ or big-model inference on our own hardware. Fastest at ~1-2s/pair via Claude API.
- **Risk:** External dependency. Vendor-lock. Cost scales with dataset size.
- **Best fit for us if:** We treat RFT as a one-shot quality pass on a modest dataset (~10k prompts × 8 completions = 80k judgments → ~$50-200 via Claude Haiku).

**Option B — Our own `confidence_head`**
- **Quality:** Unknown. `confidence_head` was designed as a training-aux signal, not a preference judge. Unclear if its scores rank completions well.
- **Cost:** Free, runs on our GPU with our model.
- **Risk:** Self-judging can produce mode collapse ("model prefers what it already produces"). Needs an A/B against Option A on a small subset before trusting it at scale.
- **Best fit for us if:** We first validate on a small pilot that its scores correlate with an external judge.

**Option C — Public preference reward model**
- **Quality:** Middle ground. Public RMs like OpenAssistant-RM, HH-RLHF-RM, PairRM.
- **Cost:** GPU inference on a separate ~1B-3B model.
- **Risk:** Distribution mismatch (public RMs trained on different data than our target domain).
- **Best fit for us if:** We want an offline, reproducible, no-API-cost pipeline.

**RECOMMENDATION:** Start with **Option A (Claude Haiku as judge)** on a *pilot* set of 500 prompts × 4 completions each (~2000 judgments, ~$5-10). If it moves the needle, scale up. Simultaneously run **Option B (confidence_head)** on the same pilot to check correlation — if ρ > 0.5 with Option A, switch to it for the full run (free + fast).

### 2. **Dataset P** — where do the prompts come from?

- **Alpaca prompts** (already have): 52k instruction prompts. Would ~repeat what SFT saw.
- **UltraChat / LMSYS Arena prompts** (open datasets): more diverse, closer to real use.
- **Self-instruct expansion**: generate more prompts from a seed set.

**RECOMMENDATION:** Use a **mix of Alpaca + UltraChat** (~30k total), so we hit both instruction-following (Alpaca) and conversational (UltraChat) distributions. Avoid using ultrafeedback prompts (SimPO already saw those; would double-count).

### 3. **N (samples per prompt) and k (keep-top)**

Literature range: N=4 to N=64, k=1 to N/2.
- N=4, k=1: cheap, but the "best" of 4 may not be much better than a random sample.
- N=16, k=4: standard mid-scale; ~10x compute vs single generation.
- N=32+: diminishing returns; used mostly in RLHF-scale runs.

**RECOMMENDATION:** **N=8, k=2** (top quartile) for the pilot. Cheap enough (~4h of 500M generation on 5090), meaningful signal.

### 4. **Sampling parameters**

- Temperature 0 gives deterministic → no diversity → RFT can't work.
- Standard: T=0.7-1.0, top_p=0.9, min_p=0.05.
- Higher T = more diverse candidates = more room for verifier to pick well, but also more noise.

**RECOMMENDATION:** T=0.8, top_p=0.9 for candidate generation. Standard defaults.

### 5. **SFT training on kept completions**

- Same optimizer/LR as our full-FT SFT (Muon 0.001 + AdamW 0.0001).
- Small token budget: kept completions × prompts × epochs. E.g., 30k prompts × 2 keeps × 200 tokens × 3 epochs = 36M tokens. Cheap (~1h at 500M).
- Question: fold this in as an *additional* SFT stage after SimPO, or restart SFT from scratch with the RFT data mixed in?

**RECOMMENDATION:** As an **additional stage after SimPO** (SFT → SimPO → RFT-SFT). Keeps the pipeline modular; lets us A/B "with RFT" vs "without" cleanly.

## Cost estimate (pilot round, ~2 days from now if we choose to run)

| Step | Cost |
|---|---|
| Generate 30k prompts × 8 completions × 200 tokens | ~4h on 5090 (500M) |
| Judge 240k completions via Claude Haiku API | $10-25 |
| SFT on top-2 (60k pairs × 3 epochs, ~36M tok) | ~1h on 5090 |
| **Total wall time** | **~5-6h** |
| **Total $ cost** | **~$10-25** |

## Open questions / risks

1. **Does RFT hurt SimPO alignment?** RFT SFT-loss could pull the model back toward "high-probability" outputs, undoing SimPO's preference calibration. Mitigation: monitor preference-acc against SimPO's original eval set before and after RFT.

2. **Does RFT preserve sparsity?** Sparsity was carefully refined in stage 1 (spike ~0.10). RFT SFT with `sparsity_lambda: 0` might let it creep back. Mitigation: monitor spike telemetry (already added to train_sft.py 2026-07-23); add a light `sparsity_lambda: 0.3` if it drifts >0.05 above SimPO endpoint.

3. **Judge cost creep.** Scaling from pilot (~$10) to full run (~$100-200) is a real number. Set a budget cap.

4. **Do we even need RFT?** SimPO already delivers preference alignment; RFT is bootstrap on top. Skip if SimPO's demo quality is already sufficient.

## Not-yet-decided (Glenn to sign off on before build)

- Verifier choice: **A (Claude API) recommended** but B (confidence_head) is cheaper
- Prompt dataset: **Alpaca+UltraChat mix recommended**
- N and k values: **N=8, k=2 for pilot recommended**
- Whether to do this at all vs. call the chain "done" after SimPO
