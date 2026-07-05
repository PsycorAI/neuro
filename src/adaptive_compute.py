"""Adaptive computation via a PonderNet-style halting head (Phase 1: dense).

Some tokens are easy (a comma after a clause) and some are hard (the first
token of a surprising continuation). A halting head lets the model spend more
refinement passes on the hard ones. This is the "adaptive computation / halting
head" item from the research agenda -- distinct from the Batch C confidence head
(which predicts correctness but drives no computation).

Phase 1 scope, deliberately narrow (mirrors the MoE Phase 1 decision):
  - A shared refinement transform is applied to the final hidden state up to
    `max_steps` times. Every step is computed for every token (DENSE), and a
    halting head produces per-token halt probabilities that combine the per-step
    states into a weighted average (PonderNet's expected output).
  - This proves whether extra refinement passes help quality at all, BEFORE
    building true dynamic early-exit (variable per-token compute graphs on top
    of the FPT-parallel recurrent kernel) -- that compute-saving payoff is the
    Phase 2 engineering, deferred until Phase 1 shows a win.

Halting math (PonderNet, Banino et al. 2021):
  step n in 0..N-1 produces halt prob lambda_n = sigmoid(halt(h_n)); the last
  step is forced (lambda_{N-1}=1). The probability of halting exactly at step n
  is  p_n = lambda_n * prod_{m<n}(1 - lambda_m). Output = sum_n p_n * h_n.
  h_0 is the unrefined input (the model can halt immediately if it's good).
"""
import torch
import torch.nn as nn

from model import _make_norm


class PonderRefine(nn.Module):
    def __init__(self, d, max_steps=3, use_rmsnorm=False):
        super().__init__()
        self.max_steps = max_steps
        self.refine = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.halt = nn.Linear(d, 1)
        self.norm = _make_norm(d, use_rmsnorm)
        # Small halt-head init so early training halts late (uses the passes)
        # rather than collapsing to "halt at step 0" before refinement can learn.
        with torch.no_grad():
            self.halt.weight.mul_(0.01)
            self.halt.bias.fill_(-2.0)   # sigmoid(-2) ~= 0.12 initial halt prob
        self._last_ponder_cost = None
        self._last_expected_steps = None

    def forward(self, x):
        """x: (B, T, d) final hidden state. Returns refined (B, T, d)."""
        B, T, _ = x.shape
        h = x
        remainder = torch.ones(B, T, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)
        p_steps = []
        for n in range(self.max_steps):
            if n > 0:
                h = self.norm(h + self.refine(h))
            if n < self.max_steps - 1:
                lam = torch.sigmoid(self.halt(h)).squeeze(-1)      # (B,T)
            else:
                lam = torch.ones(B, T, device=x.device, dtype=x.dtype)  # force halt
            p = remainder * lam                                    # halt exactly at n
            y = y + p.unsqueeze(-1) * h
            remainder = remainder * (1 - lam)
            p_steps.append(p)
        # Ponder cost = expected number of steps (ACT-style). Penalizing it in
        # the loss makes the model halt early unless the extra pass earns its
        # keep. Also the headline diagnostic: how much compute did tokens ask for?
        p_stack = torch.stack(p_steps, dim=0)                      # (N, B, T)
        step_idx = torch.arange(1, self.max_steps + 1, device=x.device).view(-1, 1, 1)
        expected_steps = (p_stack * step_idx).sum(dim=0)           # (B,T)
        self._last_ponder_cost = expected_steps.mean()
        self._last_expected_steps = expected_steps.mean().detach()
        return y
