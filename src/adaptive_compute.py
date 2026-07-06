"""Adaptive computation via Mixture-of-Recursions expert-choice routing.

Some tokens are easy (a comma after a clause) and some are hard (the first
token of a surprising continuation). We want the model to spend more refinement
passes on the hard ones. This is the "adaptive computation" research-agenda item
(distinct from the Batch C confidence head, which predicts correctness but drives
no computation).

Why this is a rewrite (2026-07-05)
----------------------------------
The first implementation was PonderNet-style (Banino et al. 2021): a per-token
halting head with an expected-steps cost penalty. It COLLAPSED. Screened at three
cost weights:
    cw=0.01 -> expected_steps 3.000 (always max)
    cw=0.03 -> expected_steps 3.000 (always max, worse ppl)
    cw=0.10 -> expected_steps 1.000 (always min)
Quality degraded monotonically with the weight; there was no stable middle. This
is the KNOWN failure mode of ACT/PonderNet halting: the mean-steps penalty has
two attractors (all-min / all-max) and the "adaptive" regime between them is a
knife-edge that a scalar weight cannot hold.

The 2025-26 literature moved past per-token halting for exactly this reason.
Mixture-of-Recursions (Bae et al., arXiv:2507.10524, Jul 2025) uses EXPERT-CHOICE
capacity routing instead: at each recursion depth a router picks the top-k tokens
(by a fixed capacity fraction) to continue; everyone else exits with their current
state. Because the capacity is a HARD top-k, exactly that fraction continues every
step -- collapse is structurally impossible, not merely discouraged by a loss. The
router only learns WHICH tokens are worth continuing, never HOW MANY (that's fixed).

Design (MoR expert-choice, adapted to Neuro's final-hidden-state refinement)
---------------------------------------------------------------------------
- A single shared refinement transform (Linear-GELU-Linear) is applied up to
  `ponder_steps - 1` times -- true recursion / a "looped" refinement, shared
  weights (the parameter-efficiency point of MoR, and literally the user's
  "looped transformer" framing).
- Depth 0 is the raw final hidden state (an easy token can exit with zero
  refinement). At each refinement step r = 1 .. S-1, a small per-depth linear
  router scores the still-active tokens; a hard top-k keeps `capacity[r]` of the
  TOTAL tokens, the rest exit with their depth-(r-1) state.
- Kept tokens get a GATED residual refinement  h_r = norm(h_{r-1} + g * refine(h_{r-1}))
  where g = sigmoid(router score). The gate scales the update so the router
  receives gradient (MoR's "the router weight both selects and scales").
- Default capacity schedule is balanced: capacity[r] = (S - r) / S, so an equal
  1/S of tokens exits at each depth (for S=3: keep 2/3 then 1/3, matching MoR).

Train vs. eval (causality)
--------------------------
Expert-choice top-k is non-causal (whether token t is kept depends on other
tokens' scores via the threshold), which would leak future information into the
teacher-forced perplexity. MoR's fix, adopted here:
  - TRAIN: hard top-k (collapse-proof) + an auxiliary BCE loss that trains each
    router's sigmoid score to predict its own top-k membership. This calibrates
    the scores to be threshold-separable.
  - EVAL: a per-token threshold on the (now-calibrated) score -- fully causal,
    so val_ppl is honest. Eval keep-rates are logged to show calibration.

Phase 1 vs Phase 2
------------------
Phase 1 (here) computes refine(h) DENSELY for all tokens and masks the result --
it proves the quality/adaptivity of the mechanism without the gather/scatter
plumbing. Phase 2 (deferred, same staging as the MoE work) gathers only the kept
tokens so the exited ones genuinely skip compute -- that's where the FLOP/energy
saving is realized. The capacity schedule already defines exactly how much compute
Phase 2 will save (for balanced S=3: mean depth 1.0 of 2 -> ~50% of refine FLOPs).

Ablation-fragility fix (2026-07-05)
------------------------------------
The first screen (`screen_mor_refine_30m`) showed memory-Δ (ablated_ppl - val_ppl)
growing to the hundreds/high-triple-digits and still climbing, vs. +tens for every
other config at this recipe/scale (dense MoE, the old dense PonderNet ponder, etc).
Researched the literature: this is TWO known, independently-documented failure
modes compounding:
  1. Discrete top-k routing is fragile to input perturbation -- MoE-robustness
     papers note that near a routing decision boundary, "even minor input
     perturbations can result in a different expert being selected." Our own
     depth_hist confirms this: it polarized over training ([0.27,0.41,0.32] ->
     [0.56,0.07,0.37]), meaning more tokens sit exactly at the fragile boundary.
  2. The refinement gate wasn't identity-safe: `g` (the router's own sigmoid
     score) was used to BOTH select top-k membership AND scale the refinement
     magnitude (h + g*refine(h)), so the tokens the router is MOST confident
     about get the STRONGEST transformation applied -- exactly backwards from
     what a graceful-degradation design wants, and initialized at ~73% strength
     from step 1 (no identity-safe warm start).

Fix (ReZero, Bachlechner et al. 2020, arXiv:2003.04887): DECOUPLE the routing
decision from the refinement magnitude. A dedicated scalar `alpha[r]` per
depth, zero-initialized, now scales the refine() update -- independent of the
router score, which is used ONLY for top-k selection + calibration. This starts
every refine step as an exact identity map (a routing flip under ablation now
changes the output by however much `alpha` has grown to trust the refinement,
not by the router's full confidence), and lets the model gradually learn how
much to invest in refinement as training shows it's warranted -- the same
mechanism ReZero uses to stabilize/speed up convergence in deep and recursive
residual stacks generally.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import _make_norm


class MoRRefine(nn.Module):
    def __init__(self, d, ponder_steps=3, capacity=None, threshold=0.5,
                 use_rmsnorm=False):
        super().__init__()
        self.ponder_steps = ponder_steps            # S: max depth (0 .. S-1)
        self.n_refine = ponder_steps - 1            # number of refine steps
        self.threshold = threshold
        # Shared refinement transform (recursion / looped), applied repeatedly.
        self.refine = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = _make_norm(d, use_rmsnorm)
        # One small linear router per refine step (MoR: linear router is best).
        self.routers = nn.ModuleList([nn.Linear(d, 1) for _ in range(self.n_refine)])
        # ReZero-style: a dedicated scalar per refine step, zero-init, DECOUPLED
        # from the router score. Controls how much the refine update perturbs h;
        # starts as an exact identity map so a routing flip (e.g. under memory
        # ablation) can't be amplified by a confidently-strong transformation --
        # see the ablation-fragility fix note above.
        self.alpha = nn.Parameter(torch.zeros(self.n_refine))
        # Balanced default capacity: keep (S-r)/S of TOTAL tokens at step r.
        if capacity is None:
            capacity = [(ponder_steps - r) / ponder_steps
                        for r in range(1, ponder_steps)]
        assert len(capacity) == self.n_refine, \
            f"capacity needs {self.n_refine} entries, got {len(capacity)}"
        self.capacity = capacity
        # Slight positive-bias init so early training keeps (refines) tokens and
        # the refine transform gets signal before the router learns to be picky.
        for r in self.routers:
            with torch.no_grad():
                r.bias.fill_(1.0)
        self._last_ponder_aux = None
        self._last_expected_steps = None
        self._last_depth_hist = None
        self._last_eval_keep = None

    def forward(self, x):
        """x: (B, T, d) final hidden state. Returns refined (B, T, d)."""
        B, T, D = x.shape
        N = B * T
        h = x.reshape(N, D)
        output = h.clone()                                  # default: exit depth 0
        active = torch.ones(N, dtype=torch.bool, device=x.device)
        depth = torch.zeros(N, device=x.device)             # refinements received
        aux = x.new_zeros(())
        keep_rates = []

        for r in range(self.n_refine):
            cap = self.capacity[r]
            score = self.routers[r](h).squeeze(-1)                  # (N,) raw logit
            g = torch.sigmoid(score)

            if self.training:
                # Hard top-k over ACTIVE tokens -> collapse-proof by construction.
                k = max(1, int(round(cap * N)))
                sel = torch.where(active, g, g.new_full((), -1.0).expand(N))
                topk_idx = torch.topk(sel, min(k, N)).indices
                keep = torch.zeros(N, dtype=torch.bool, device=x.device)
                keep[topk_idx] = True
                keep &= active
                # Aux BCE: calibrate score -> top-k membership (causal-eval bridge).
                # Uses logits (not the sigmoid output) -- with_logits is autocast-safe,
                # plain binary_cross_entropy on a sigmoid'd tensor is NOT (raises
                # under torch.autocast/bf16 mixed precision).
                if active.any():
                    aux = aux + F.binary_cross_entropy_with_logits(
                        score[active], keep[active].float().detach())
            else:
                # Causal per-token threshold (no cross-token leak -> honest ppl).
                keep = active & (g > self.threshold)
                keep_rates.append(keep.float().mean())

            # ReZero-decoupled residual refinement (dense compute in Phase 1,
            # masked by keep). alpha (not g) scales the update -- the routing
            # decision and the refinement magnitude are independent.
            refined = self.norm(h + self.alpha[r] * self.refine(h))
            keep_c = keep.unsqueeze(-1)
            h = torch.where(keep_c, refined, h)
            output = torch.where(keep_c, refined, output)
            depth = depth + keep.float()
            active = keep

        self._last_ponder_aux = aux
        self._last_expected_steps = depth.mean().detach()
        self._last_depth_hist = torch.stack(
            [(depth == dd).float().mean() for dd in range(self.ponder_steps)]).detach()
        self._last_eval_keep = (torch.stack(keep_rates).detach()
                                if keep_rates else None)
        # How much the model has learned to trust the refinement -- should grow
        # away from 0 during training if the mechanism is earning its keep, not
        # just sitting at identity to dodge the ablation-fragility issue.
        self._last_alpha = self.alpha.detach().clone()
        return output.reshape(B, T, D)
