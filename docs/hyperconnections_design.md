# Hyper-connections for Neuro — design & 350M decision note

**Status:** design only, not built. Decision gate: ablate at 30M **after** the
NorMuon A/B lands, bake into the 350M run only if it clears a clean A/B.
**Refs:** Hyper-Connections (ByteDance, arXiv:2409.19606) · mHC-lite
(2601.05732) · Frac-connections (2503.14125).

---

## 1. What it is

A drop-in generalization of the residual connection. Instead of one residual
stream `x = x + block(x)`, carry **n parallel streams** and let the network learn
how to *mix across streams* (width connections) and *how much of each stream feeds
each block* (depth connections). The static residual is the special case n=1 with
the connection matrix pinned to identity.

Two cheaper variants matter more to us than the full method:

- **Frac-connections (2503.14125):** don't widen the stream n×; instead *split*
  the existing d-dim stream into n fractions of width d/n and connect those.
  Same memory as a plain residual, a handful of extra scalars/vectors. This is
  the variant to try first — it gets most of the reported stability/quality win
  at ~zero memory cost.
- **mHC-lite (2601.05732):** a parameter-light hyper-connection (shared/low-rank
  connection weights) — middle ground if Frac underwhelms.

Reported benefit (transformers): faster, more stable convergence and a small
quality gain, architecture-agnostic, minimal slowdown. Nothing about it is
attention-specific, so it is compatible with our non-attention thesis.

## 2. Where it plugs into Neuro

Today the block stack is a single residual stream. In `src/model.py` the add
appears in **three** mirrored places (the plain forward and the two
recurrent-depth/stateful paths):

```python
# _run_blocks (line ~707), forward stateful (~848), and (~904):
out_seq = block.forward_sequence(x_seq, ...)
if L > 1 and (l > 0 or use_residual):
    out_seq = out_seq + x_seq       # <-- single-stream residual
x_seq = out_seq
```

A hyper/Frac-connection replaces exactly this add. For Frac-connections the
per-layer state becomes a small set of learnable mix coefficients over the n
fractions; the block still sees a d-dim input (fractions are concatenated back),
so **nothing inside the block changes** — `forward_sequence`, the delta rule, the
FLA kernel, MoR router, sparsity, β-floor all stay byte-identical. Only the
inter-block plumbing changes.

Constraints already in the code that the design must respect:
- **`use_residual = (d == d_mem)`** and the `n_layers > 1` guard (line 526): the
  residual only exists when `d == d_mem`. Frac-connections inherit the same
  requirement (they operate on the residual stream), so the flag must no-op when
  `d != d_mem`, exactly like the current add.
- **Depth-scaled init** (line ~541, `1/sqrt(2*n_layers)` on residual-adjacent
  weights) is tuned for a single stream. With n fractions the variance budget
  changes; the Frac mix coefficients must be initialized so that **at step 0 the
  module is an exact identity residual** (same discipline as Engram v2's zero-init
  proj and LoRA's zero-init B) — otherwise it perturbs a regime we've already
  tuned. This is the single most important init detail.
- **Recurrent-depth / MoR:** the router loop re-enters blocks; the connection
  state is per-*layer*, not per-*depth-iteration*, so it composes cleanly — but
  the A/B must be run with `recurrent_depth=2` + router enabled (the 350M setting)
  so the interaction is actually measured, not just the depth-1 case.

## 3. Cost

At n=2 fractions, 2 layers (30M) / 4 layers (90M):
- **Params:** negligible — O(n) to O(n²) scalars per layer for Frac/mHC-lite
  (tens to low-hundreds of floats total). Full hyper-connections with per-channel
  weights would be larger; not proposed.
- **Memory:** Frac-connections = same as plain residual (no stream widening).
  Full hyper-connections = n× the residual activation; **rejected** on the same
  16GB-headroom grounds that killed `vector_beta`.
- **Compute:** a few extra elementwise ops per block boundary; ~1–3% step-time,
  consistent with the paper.

## 4. Risks specific to us

1. **Init/variance interaction with depth-scaled init** (§2) — the main one.
   Mitigation: exact-identity init + verify residual-stream variance is unchanged
   at step 0 with a CPU probe before any GPU run.
2. **Three mirrored call sites** must stay in lockstep — a Frac add applied in the
   plain path but missed in a stateful path would silently diverge train vs
   eval/recall. Mitigation: factor the connection into one helper called by all
   three sites (small, additive, behind a flag — matches
   `feedback_live_run_code_safety`).
3. **Confound with the β-fix / MoR / NorMuon stack.** We are stacking several
   changes into one 350M run with no baseline. Hyper-connections is the *least*
   proven of them. It must earn its place with its own isolated A/B, or it is
   noise we can't attribute later.

## 5. Decision gate & A/B plan

**Do not build speculatively.** Sequence:

1. Land the NorMuon A/B first (already queued, configs ready). It's the
   cheapest/most-proven win and clears the 5080.
2. **Then** build Frac-connections behind `fracconn_enabled: false`, single helper,
   exact-identity init, CPU variance probe (same test pattern as `src/lora.py`
   and `src/engram_v2.py`).
3. 30M A/B, `recurrent_depth=2` + router on to match the 350M setting, `beta_floor:
   0.1` baked in, seed-matched control vs treatment, ~100M tokens/arm (the standard
   screen budget on the 5080). Success = a clean val_ppl improvement (or equal ppl
   at fewer tokens) that survives a seed check — not a single-seed blip.
4. Only if it clears **both** the 30M A/B and a 90M confirm does it go into the
   350M recipe. Given the "every enhancement before the 11-day run" goal, budget
   for the 90M confirm — an unproven architectural change baked into an 11-day run
   with no baseline is the exact failure mode this note exists to prevent.

## 6. Recommendation

Frac-connections (2503.14125) is the right first cut: near-zero cost, respects the
`d == d_mem` residual, and the exact-identity init makes it restart-safe to drop
into any checkpoint. Worth building and A/B-ing **after NorMuon**, ahead of the
350M. Full hyper-connections is rejected (memory), mHC-lite is the fallback only if
Frac underwhelms.
