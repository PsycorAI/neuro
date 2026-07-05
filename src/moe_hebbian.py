"""Hebbian Mixture-of-Experts, Phase 1 (dense/soft routing).

N parallel Hebbian expert lanes per layer, each with its own full-size M-state
and own W_k/W_v/W_q/W_ff, routed by a shared spike pattern. Every expert is
computed for every token (dense); the output is a router-weighted blend. This
proves whether expert specialization helps quality before building the sparse
gather/scatter path (Phase 2) that would make it compute-cheap.

Design notes:
  - The LIF spiking front-end (to_current + lif_parallel) is SHARED across
    experts -- only the memory+FF path is replicated. This mirrors standard
    MoE practice (replicate the FFN, not the attention/routing substrate) and
    literally matches "routing by spike pattern": the router sees the same
    spikes that feed every expert, no chicken-and-egg problem.
  - Expert dimension is collapsed into the batch dimension for the memory
    update, exactly like SpikingHebbianBlock's n_heads mechanism -- so the
    existing delta_chunked_gated / delta_fla_gated kernels are reused
    UNMODIFIED. The difference from n_heads: each expert gets the FULL d_mem
    (real per-expert capacity), not a split fraction of it.
  - Scope is intentionally narrow: only the FPT/chunked path, and always
    gated-delta (decay_gate + delta_rule) -- the one path every current
    production config actually uses. Recurrent/titans/local_attn/pre_conv/
    vector_beta are out of scope for Phase 1.
"""
import torch
import torch.nn as nn

from model import SwiGLUFF, _make_norm


class MoEHebbianBlock(nn.Module):
    def __init__(self, d_in, n_neurons, d_mem, n_experts, beta=0.9,
                 beta_floor=0.0, use_rmsnorm=False, use_swiglu=False,
                 depth_scale=1.0, compile_inv=False, use_fla_kernel=False,
                 route_topk=None, route_noise_std=0.0, router_z_weight=0.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_mem = d_mem
        self.n_experts = n_experts
        # Tweaks (default-off): route_noise_std adds Gaussian noise to router
        # logits during training (Shazeer noisy-top-k -- encourages exploration
        # / better load balance). router_z_weight penalizes large router logits
        # (ST-MoE router z-loss -- keeps the gate softmax numerically calm).
        self.route_noise_std = route_noise_std
        self.router_z_weight = router_z_weight
        # route_topk: keep only top-k experts per token in the READ blend.
        # None or >= n_experts -> dense soft (current default). k=1 = hard.
        # Note: memory WRITE is still dense (all experts update M) -- this is
        # Phase 1.5 (tests whether hard READ helps quality). True sparse compute
        # with per-expert token subsequences is Phase 2 engineering.
        self.route_topk = route_topk if route_topk and route_topk < n_experts else None
        # LM-level state allocator reads these (mirrors n_heads/d_head on the
        # single-expert block) to size the carried M-state correctly.
        self.n_state_units = n_experts
        self.d_state = d_mem
        self.beta_val = beta
        self.beta_floor = beta_floor
        self.compile_inv = compile_inv
        self.use_fla_kernel = use_fla_kernel
        self.lam = 0.98   # unused (gated-delta forget is data-dependent); kept
                           # only so _set_lam's blanket loop over blocks doesn't break.

        self.to_current = nn.Linear(d_in, n_neurons)
        self.router = nn.Linear(n_neurons, n_experts)

        E, D = n_experts, d_mem
        self.W_k = nn.Linear(n_neurons, E * D, bias=False)
        self.W_v = nn.Linear(n_neurons, E * D, bias=False)
        self.W_q = nn.Linear(n_neurons, E * D, bias=False)
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.W_ff = SwiGLUFF(n_neurons, E * D, bias=False)
        else:
            self.W_ff = nn.Linear(n_neurons, E * D, bias=False)
        self.W_beta = nn.Linear(n_neurons, E)
        self.W_alpha = nn.Linear(n_neurons, E)
        with torch.no_grad():
            # Same init as SpikingHebbianBlock's decay_gate: bias high so
            # initial alpha ~= 0.98, model learns to forget faster where useful.
            self.W_alpha.weight.mul_(0.01)
            self.W_alpha.bias.fill_(4.0)

        self.norm = _make_norm(d_mem, use_rmsnorm)

        if depth_scale != 1.0:
            out_lin = self.W_ff.down if use_swiglu else self.W_ff
            with torch.no_grad():
                out_lin.weight.mul_(depth_scale)

        self._last_aux_loss = None

    def forward_sequence(self, x_seq, mem, M, prev_spk, ablate_memory=False,
                         fpt_K=10):
        from lif_parallel import lif_parallel
        B, T, _ = x_seq.shape
        E, D = self.n_experts, self.d_mem

        cur_seq = self.to_current(x_seq)
        spk_seq, mem_final = lif_parallel(cur_seq, self.beta_val, vth=1.0, K=fpt_K)
        prev_seq = torch.cat([prev_spk.unsqueeze(1), spk_seq[:, :-1, :]], dim=1)

        router_logits = self.router(spk_seq)                     # (B,T,E)
        if self.training and self.route_noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.route_noise_std
        route_w_soft = torch.softmax(router_logits, dim=-1)      # for aux loss
        if self.route_topk is not None:
            # Top-k mask + renormalize. Gradient still flows through the k
            # surviving weights; non-selected experts get zero blend-path
            # gradient but continue to receive gradient via the aux loss
            # (which uses the full soft distribution). No STE needed.
            topk_vals, topk_idx = router_logits.topk(self.route_topk, dim=-1)
            mask = torch.zeros_like(router_logits).scatter_(-1, topk_idx, 1.0)
            gated = route_w_soft * mask
            route_w = gated / (gated.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            route_w = route_w_soft

        k_seq = self.W_k(prev_seq).view(B, T, E, D)
        v_seq = self.W_v(spk_seq).view(B, T, E, D)
        q_seq = self.W_q(spk_seq).view(B, T, E, D)
        ff_seq = self.W_ff(spk_seq).view(B, T, E, D)
        beta_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk_seq))  # (B,T,E)
        alpha_seq = torch.sigmoid(self.W_alpha(spk_seq)).clamp(0.5, 0.9999)                        # (B,T,E)

        if not ablate_memory:
            # Collapse expert dim into batch -- same trick as n_heads>1, just
            # with full D per lane instead of a split fraction.
            def collapse(t):
                return t.permute(0, 2, 1, 3).reshape(B * E, T, D)
            v_in, k_in, q_in = collapse(v_seq), collapse(k_seq), collapse(q_seq)
            beta_in = beta_seq.permute(0, 2, 1).reshape(B * E, T, 1)
            alpha_in = alpha_seq.permute(0, 2, 1).reshape(B * E, T, 1)
            if self.use_fla_kernel:
                from delta_chunked import delta_fla_gated
                r_in, M = delta_fla_gated(v_in, k_in, q_in, beta_in, alpha_in, M)
            else:
                from delta_chunked import delta_chunked_gated
                r_in, M = delta_chunked_gated(v_in, k_in, q_in, beta_in, alpha_in, M,
                                               chunk=64, compile_inv=self.compile_inv)
            r_seq = r_in.reshape(B, E, T, D).permute(0, 2, 1, 3)  # (B,T,E,D)
        else:
            r_seq = torch.zeros_like(ff_seq)

        w = route_w.unsqueeze(-1)                                 # (B,T,E,1)
        blended = ((r_seq + ff_seq) * w).sum(dim=2)                # (B,T,D)
        out_seq = self.norm(blended)

        # Load-balance aux loss (Switch-Transformer importance loss): penalize
        # uneven total router mass across experts. Without this, spiking
        # routers collapse onto one expert fast. Use the SOFT distribution
        # (pre top-k) so gradient reaches non-selected experts too -- otherwise
        # top-1 routing can permanently starve unused experts.
        importance = route_w_soft.reshape(-1, E).sum(dim=0)        # (E,)
        cv = importance.std() / (importance.mean() + 1e-9)
        aux = cv * cv
        if self.router_z_weight > 0:
            # ST-MoE router z-loss: mean of (logsumexp of router logits)^2.
            z = torch.logsumexp(router_logits, dim=-1)             # (B,T)
            aux = aux + self.router_z_weight * (z * z).mean()
        self._last_aux_loss = aux
        # Diagnostic only (not used in the loss): mean routing weight per
        # expert, to tell "balanced load" (aux loss low) apart from "genuine
        # per-token specialization" (route_w peaky per token, not just uniform
        # on average).
        self._last_route_util = route_w.reshape(-1, E).mean(dim=0).detach()
        self._last_route_peak = route_w.detach().max(dim=-1).values.mean()

        final_spk = spk_seq[:, -1, :]
        return out_seq, mem_final, M, spk_seq, final_spk
