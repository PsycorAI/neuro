# EBM-lite (energy/verifier head) — design & scoping note

**Date:** 2026-07-10
**Status:** scoping only, not built. Explicitly gated "after MoR resolves"
(to-do idea B). This note exists to decide *whether* it's worth building and,
if so, to pin the one thing this project keeps getting burned on: a crisp
success criterion.
**Not:** full equilibrium-prop / energy-based *training* of the whole model —
that would rewrite the backprop+Muon+KD pipeline and is rejected as impractical.
This is a small auxiliary **verifier head**, trained contrastively, used to
rerank/reject candidate continuations at inference.

---

## 1. What it is (honest scope)

A scalar head `E(context, candidate) → ℝ` where **low energy = coherent/likely**
continuation, high energy = incoherent. Trained contrastively: real
continuations get pushed low, "bad" continuations get pushed high. At inference
it's a **reranker** over N candidate continuations (or a **reject** threshold),
targeting hallucination / local incoherence / self-contradiction — failure modes
that token-level cross-entropy doesn't directly penalize.

It is NOT a preference/reward model (no human labels, not RLHF) and NOT a second
LM. It's one small head plus a contrastive loss.

## 2. Why now is good for *scoping*, not *building*

- **Gated behind MoR.** MoR is mid-resolution (90M β-fix confirming, 350M
  decision open). The gate is real: EBM's value depends on a settled base.
- **Post-pretrain / inference-time.** Like LoRA, it attaches on top of a trained
  base and does NOT need to be in the 350M pretrain. The "bake everything into
  the 11-day run" urgency does not apply.
- **Most speculative item in the backlog**, and its eval is the hard part (§4).
  Building before the eval is designed is how you get another MoE/Engram-style
  ambiguous-against-noise result.

## 3. Neuro-specific design (if built)

**Attach point.** Mirror the existing `confidence_head` (`nn.Linear(d_mem, 1)` on
the final `x_seq`, model.py:625). The reference to port is **EORM**
(arXiv:2505.14999, code: github.com/ericjiang18/EnergyORM) — a 55M energy
reranker, our exact use case and scale. Its energy head is literally:

```
E_θ(y) = MLP(LayerNorm(h_pool))  ∈ ℝ        # lower energy = better
```

For Neuro: run the base over `[context; candidate]`, pool the candidate-span
hiddens of `x_seq` (EORM prepends a [CLS] and reads its final hidden; we're
causal, so the **last-token hidden of the candidate span** is the natural analog,
or mean-pool the span), then `LayerNorm → 2-layer MLP → 1`. Two architecture
options, decide at build:
  - **Head-on-base (cheap, first cut):** reuse the frozen base's hiddens, train
    only the head. ~free, but the base is causal and tuned for next-token pred,
    so its representation may be a weak judge — if head-only underperforms, add
    LoRA on the base (reuses `src/lora.py`) before concluding EBM doesn't work.
  - **Standalone verifier (EORM-style):** a small independent encoder over the
    candidate text (EORM: GPT-2 tokenizer, bidirectional, [CLS]-pooled, 55M).
    More judging capacity + a bidirectional read of the whole candidate, but a
    separate model to train/host. Only go here if head-on-base + LoRA is weak.

**Shortcut worth knowing (EORM Prop. C.1):** a binary correct/incorrect
classifier's logit IS a negative energy — `E = −logit`. So the existing
`confidence_head` can be repurposed almost as-is: train it as a
correct/incorrect discriminator with the pairwise loss below and read `−logit`
as the energy. This is the minimal-code path and the one to try first.

**Training signal — the negatives are the whole ballgame.** Positives = the
true continuation. Negative sources, cheapest→richest:
  1. **Self-samples**: continuations sampled from the base LM itself (temperature
     > 0). Cheap, and directly targets "things the model would say but shouldn't."
  2. **Mismatched**: a real continuation from a *different* context (tests
     context-conditioning, not just fluency).
  3. **Corrupted**: truncate/shuffle/token-swap a real continuation (tests local
     coherence). Weakest signal — easy to game, use sparingly.
Loss: **Bradley–Terry pairwise, grouped per problem** (EORM's exact formulation).
For each context `n`, partition its candidates into correct `Y₊` and incorrect
`Y₋` and average the pairwise term over the full `Y₊ × Y₋` product:

```
ℒ(θ; n) = 1/(|Y₊||Y₋|) · Σ_{y₊∈Y₊} Σ_{y₋∈Y₋} log(1 + exp(E(y₊) − E(y₋)))
```

Simpler and better-behaved than a hard margin (distributional EBM,
arXiv:2605.18871; EORM, arXiv:2505.14999). **Split at the problem/context level,
never within** — same-context pairs across train and eval is the leakage that
manufactures a fake win. Start with self-samples + mismatched, ~1:4 pos:neg.

**Data recipe (EORM template).** Per problem, sample N candidates from the
base(s) at temperature ~0.7 (EORM: N=256 in-distribution, from 5 different 7–8B
models; for us, self-samples from the 500M base are the one-model version),
assign binary outcome labels (correct iff final answer matches ground truth),
group by problem. Inference is `y* = argmin_y E(y)` over the candidate pool.

**Synergy — the strongest reason to want this in *Neuro specifically*:**
  - **MTP self-speculative decoding** (`scripts/spec_decode.py`) already drafts
    multiple continuations. An energy head is the natural **reranker** over those
    drafts — a learned scorer instead of just the base LM's own logprob.
  - **Self-refine** (draft/critique/revise with Hebbian state) currently critiques
    heuristically. E(context, draft) vs E(context, revision) gives the critique
    step a **learned signal** — arguably the cleanest place EBM earns its keep.

## 4. The eval — make-or-break (design this BEFORE building)

The project's recurring failure: a striking result that's really just
15–20-ppl / seed-level noise (MoE, Engram v1, uniform recurrent-depth). EBM has
NO natural scalar like val_ppl, so this risk is *higher*, not lower. The 2026
literature (distributional EBM, arXiv:2605.18871) converged on a standard
protocol — adopt it rather than invent one.

**Primary eval — pass@N reranking curve (this is the field-standard gold metric).**
Build a candidate pool per prompt (e.g. 32 = 8 samples × 4 generators, or 8× the
base at T>0 if only one generator). Rerank by E, plot pass@N of the reranked
order. Report against three references on the SAME pool:
  - **oracle@N** (upper bound = headroom; if greedy≈oracle there's nothing to
    rerank, wrong task),
  - **self-consistency majority vote** and **greedy@T=0** (the free baselines),
  - **base-LM logprob ranking** (the "does the head add anything over the model's
    own probability" test — the one that actually matters).
Pre-register the win: energy-reranked pass@N must clear base-logprob pass@N by a
margin on held-out prompts. If a task's greedy is already near-oracle, drop it.

**Required confound battery (skipping this is how we'd fake a win).** A high
pass@N can come from the scorer learning *generator identity* or *formatting*, not
quality. Before trusting any result, run all three (all from arXiv:2605.18871):
  1. **Cross-model energy spread** — energy variance across generators for the
     same content should be LOW (content-based, not fingerprinting the source).
  2. **Per-generator pick distribution** — reranker's picks ~uniform across
     generators, not always-from-one.
  3. **Style-swap test** — reformat a correct answer into a "bad-looking" style;
     if E flips, it learned formatting shortcuts, not correctness.

**Secondary — selective prediction.** Treat energy (or an ensemble σ) as
uncertainty: AUROC for correct-vs-incorrect separation, and a risk–coverage
curve with abstention capped (~<10%). Demo-friendly ("the model knows when it
doesn't know") and reuses the same scores.

If the primary can't be made crisp with data on hand, or a win doesn't survive
the confound battery, **that is the verdict: defer EBM** — the idea isn't wrong,
it's that we can't measure it cleanly, and an unmeasurable exploratory head is
exactly what not to spend a GPU window on.

## 5. Cost & risk

- **Params/compute:** tiny head (one MLP → scalar). The cost is the
  negatives-generation (sampling from the base) and the eval harness, not the
  model.
- **Risk:** low blast radius (additive, behind an `ebm_enabled` flag, frozen base
  option = can't hurt the base). The real risk is *wasted effort* on an
  ambiguous result — mitigated entirely by §4.

## 6. Recommendation & decision gate

1. **Now:** this note. No code.
2. **After** the 90M β-fix confirms AND the NorMuon/Engram v2 A/Bs run (they're
   more-proven and already prepped — EBM should not jump the queue).
3. **Before any EBM code:** commit to eval #1 or #2 from §4 as the pre-registered
   win condition. If we can't, defer and revisit post-500M-demo.
4. **First build (if it clears the gate):** frozen-base head + self-sample /
   mismatched negatives + ranking-accuracy eval (§4.1) at 30M or on the 500M
   base. Only chase best-of-N lift (§4.2) if ranking accuracy clears its
   threshold — that ordering fails fast and cheap.

**Best home in the stack:** as the learned scorer for self-refine and/or MTP
spec-decode reranking — that's where a coherence verifier turns into a visible
demo capability, and it reuses machinery that already exists.

---

## References

**Port this one (closest to our build):**
- **EORM** — "Learning to Rank Chain-of-Thought: An Energy-Based Approach with
  Outcome Supervision" (arXiv:2505.14999, 2025). A 55M energy reranker: head =
  `MLP(LayerNorm(h_CLS))`, per-problem Bradley–Terry loss, outcome-label
  supervision, `argmin E` inference. Our scale, our task. **Code:**
  https://github.com/ericjiang18/EnergyORM — the reference implementation to port
  the head + loss + candidate-pool pipeline from.

**Eval protocol & rigor:**
- Distributional EBM for uncertainty-aware structured LLM reasoning
  (arXiv:2605.18871, 2026) — source of §4's eval protocol: pass@N reranking vs
  oracle/self-consistency/base-logprob, and the cross-model / per-generator /
  style-swap confound battery. Also energy-as-ensemble-σ for selective
  prediction (AUROC / risk–coverage).

**Method lineage & related patterns:**
- Energy-Based Reranking for NMT (arXiv:2009.13267) — the original
  energy-reranks-a-candidate-pool recipe; long-established, working pattern.
- QE-EBM (arXiv:2410.10228) — quality estimator used as an energy loss; useful
  if we ever want the energy to also shape generation, not just rerank.
- CoRe-Reranking (arXiv:2510.02219, code: github.com/linhhtran/CoRe-Reranking) —
  decoder-hidden mean-pool + binary head reranker; a clean reference for the
  head-on-base pooling variant (§3, option 1).
- autoregressive LMs are themselves EBMs (arXiv 2025) — the base's own logprob IS
  an energy, which is exactly why "beat base-logprob ranking" is the load-bearing
  baseline in §4, not a nice-to-have.
