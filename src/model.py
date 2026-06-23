"""Phase 1+: Spiking Hebbian language model.

Token model where:
  - Leaky integrate-and-fire (LIF) layers emit sparse binary spikes (the "neurons")
  - Decaying Hebbian fast-weight matrices M store outer-product associations (the "synapses")
  - M binds the PREVIOUS token's key to the CURRENT token's value, so a later
    occurrence of a token can retrieve its learned successor (induction / recall).

      M_t = lambda * M_{t-1} + eta * (v_t  outer  k_{t-1})
      r_t = M_t q_t

Multi-layer: n_layers blocks are stacked with residual connections (requires d == d_mem
for n_layers > 1). Each layer has its own LIF state, Hebbian memory M, and optionally
SET recurrent connections.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import Leaky, surrogate


class SpikingHebbianBlock(nn.Module):
    """Single spiking-Hebbian layer."""

    def __init__(self, d_in, n_neurons, d_mem, beta=0.9, lam=0.98, eta=1.0,
                 compile_safe=False, recurrent=False, rec_density=0.05,
                 learnable_decay=False, write_gate=False, delta_rule=False,
                 beta_floor=0.0, decay_gate=False, titans=False,
                 local_attn=False, local_window=64,
                 n_heads=1, pre_conv=False, pre_conv_kernel=4,
                 vector_beta=False):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_mem = d_mem
        self.n_heads = n_heads
        if d_mem % n_heads != 0:
            raise ValueError(f"d_mem ({d_mem}) must be divisible by n_heads ({n_heads})")
        self.d_head = d_mem // n_heads
        self.lam = lam
        self.eta = eta
        self.compile_safe = compile_safe
        self.beta_val = beta
        self.learnable_decay = learnable_decay
        self.write_gate = write_gate
        self.delta_rule = delta_rule
        self.beta_floor = beta_floor   # min write strength: β = floor + (1-floor)·σ(W_β)
        self.decay_gate = decay_gate   # Gated DeltaNet: data-dependent forget α = σ(W_alpha(spk))
        # ST Phase 8: delta-rule write strength β = sigmoid(W_beta(spk)).
        # vector_beta (RWKV-7 style): per-channel β instead of scalar — write
        # strength can differ across memory-output dimensions.
        self.vector_beta = vector_beta
        if delta_rule:
            self.W_beta = nn.Linear(n_neurons, d_mem if vector_beta else 1)
        if decay_gate:
            # Init bias high so initial α ≈ 0.98 (matches default lam) — model can
            # then learn to LOWER α (forget faster) where useful.
            self.W_alpha = nn.Linear(n_neurons, 1)
            with torch.no_grad():
                self.W_alpha.weight.mul_(0.01)
                self.W_alpha.bias.fill_(4.0)        # sigmoid(4.0) ≈ 0.982
        self.titans = titans
        if titans:
            # Titans: gradient-descent memory with momentum + data-dependent forget.
            # θ (write strength / learning rate), η (momentum decay), α (forget gate).
            # Init: θ small (gentle write), η high (preserve momentum), α small (mostly keep).
            self.W_theta = nn.Linear(n_neurons, 1)
            self.W_eta   = nn.Linear(n_neurons, 1)
            self.W_alpha_t = nn.Linear(n_neurons, 1)
            with torch.no_grad():
                for L_, b in ((self.W_theta, -2.0), (self.W_eta, 2.0), (self.W_alpha_t, -4.0)):
                    L_.weight.mul_(0.01); L_.bias.fill_(b)
            # Titans also needs k,v,q projections (already created below).
        self.local_attn = local_attn
        self.local_window = local_window
        if local_attn:
            # BASED-style: tiny causal sliding-window exact attention added to the
            # memory read. Reuses W_k/W_v/W_q. A learnable per-block scalar gate so
            # the model can pick how much local-recall help it wants (init 0 -> no-op
            # at training start; weights gain ground gradually).
            self.local_gate = nn.Parameter(torch.zeros(1))
        self.pre_conv = pre_conv
        self.pre_conv_kernel = pre_conv_kernel
        if pre_conv:
            # Per 2508.19029: causal depthwise 1D conv on spike sequence before
            # the W_k/W_v/W_q projections. Improves recall (mixes neighbor timesteps
            # so the projection sees a local window). Depthwise (groups=N) = cheap.
            self.spk_conv = nn.Conv1d(n_neurons, n_neurons,
                                       kernel_size=pre_conv_kernel,
                                       padding=0, groups=n_neurons)

        # ST Phase 4: per-(value-dim) learnable decay. alpha = sigmoid(raw).
        # Init MULTI-SCALE (RetNet-style): spread alpha across dims so the memory
        # has many timescales from the start and each dim gets distinct gradient
        # signal (the clumped alpha=0.99 init never differentiated). 1-alpha is
        # geometric from 0.001 (alpha 0.999, long memory) to 0.10 (alpha 0.90, short).
        if learnable_decay:
            one_minus = torch.logspace(math.log10(0.001), math.log10(0.10), d_mem)
            alphas = (1.0 - one_minus).clamp(0.5, 0.9999)
            self.decay_raw = nn.Parameter(torch.logit(alphas))
        # ST Phase 5: learnable scalar write gate g_t = sigmoid(W_gate(spk)).
        if write_gate:
            self.W_gate = nn.Linear(n_neurons, 1)

        self.to_current = nn.Linear(d_in, n_neurons)
        if not compile_safe:
            self.lif = Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        if recurrent:
            self.W_rec = nn.Linear(n_neurons, n_neurons, bias=False)
            mask = (torch.rand(n_neurons, n_neurons) < rec_density).float()
            mask.fill_diagonal_(0)
            self.register_buffer("rec_mask", mask)
        self.recurrent = recurrent
        self.W_k = nn.Linear(n_neurons, d_mem, bias=False)
        self.W_v = nn.Linear(n_neurons, d_mem, bias=False)
        self.W_q = nn.Linear(n_neurons, d_mem, bias=False)
        self.W_ff = nn.Linear(n_neurons, d_mem, bias=False)
        self.norm = nn.LayerNorm(d_mem)

    def _lif_step(self, cur, mem):
        """Inline LIF (vth=1, reset-by-subtraction, atan surrogate). Compile-friendly."""
        mem = self.beta_val * mem + cur
        over = mem - 1.0
        spk_hard = (over > 0).float()
        alpha = 2.0
        g = (1.0 / (math.pi * alpha)) * torch.atan(math.pi * alpha * over)
        spk = spk_hard.detach() + g - g.detach()
        mem = mem - spk_hard
        return spk, mem

    def step(self, x, mem, M, prev_spk, ablate_memory=False):
        """Process one timestep. Returns (output, mem, M, spk)."""
        cur = self.to_current(x)
        if self.recurrent:
            cur = cur + F.linear(prev_spk, self.W_rec.weight * self.rec_mask)
        if self.compile_safe:
            spk, mem = self._lif_step(cur, mem)
        else:
            spk, mem = self.lif(cur, mem)
        k = self.W_k(prev_spk)
        v = self.W_v(spk)
        q = self.W_q(spk)
        ff = self.W_ff(spk)
        if self.write_gate:
            v = v * torch.sigmoid(self.W_gate(spk))          # selective write
        if not ablate_memory:
            if self.learnable_decay:
                a = torch.sigmoid(self.decay_raw).clamp(0.5, 0.9999).view(1, -1, 1)
            if self.delta_rule:
                # Gated DeltaNet: M = αM + β(v − αM·k̂)⊗k̂  (error-driven write).
                # k̂ = L2-normalized key; β = learned input-dependent write strength
                # (scalar (B,1) or per-channel vector (B,D) when vector_beta=True);
                # α = learned forget gate (decay_gate=True) or scalar lam.
                kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
                beta = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk))  # (B,1) or (B,D)
                if self.decay_gate:
                    a_dyn = torch.sigmoid(self.W_alpha(spk)).clamp(0.5, 0.9999).unsqueeze(2)
                    prevM = a_dyn * M
                elif self.learnable_decay:
                    prevM = a * M
                else:
                    prevM = self.lam * M
                Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)  # λM·k̂  (B,dm)
                delta = (beta * (v - Mk)).unsqueeze(2)            # (B,dm,1)
                M = prevM + torch.bmm(delta, kn.unsqueeze(1))
            elif self.learnable_decay:
                M = a * M + self.eta * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            else:
                M = self.lam * M + self.eta * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            r = torch.bmm(M, q.unsqueeze(2)).squeeze(2)
        else:
            r = torch.zeros_like(ff)
        out = self.norm(r + ff)
        return out, mem, M, spk

    def step_titans(self, x, mem, M, S, prev_spk, ablate_memory=False):
        """Titans one-step update: gradient-descent memory with momentum + forget.
            e = M k̂ − v ; S ← η S − θ e k̂ᵀ ; M ← (1−α) M + S ; r = M q
        """
        cur = self.to_current(x)
        if self.recurrent:
            cur = cur + torch.nn.functional.linear(prev_spk, self.W_rec.weight * self.rec_mask)
        if self.compile_safe:
            spk, mem = self._lif_step(cur, mem)
        else:
            spk, mem = self.lif(cur, mem)
        k = self.W_k(prev_spk); v = self.W_v(spk); q = self.W_q(spk); ff = self.W_ff(spk)
        if self.write_gate:
            v = v * torch.sigmoid(self.W_gate(spk))
        if not ablate_memory:
            kn = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            th = torch.sigmoid(self.W_theta(spk)).unsqueeze(2)        # (B,1,1) learning rate
            et = torch.sigmoid(self.W_eta(spk)).unsqueeze(2)          # momentum decay
            al = torch.sigmoid(self.W_alpha_t(spk)).unsqueeze(2)      # forget gate
            Mk = torch.bmm(M, kn.unsqueeze(2)).squeeze(2)
            e = Mk - v                                                 # surprise (B,D)
            grad = torch.bmm(e.unsqueeze(2), kn.unsqueeze(1))         # (B,D,D)
            S = et * S - th * grad
            M = (1 - al) * M + S
            r = torch.bmm(M, q.unsqueeze(2)).squeeze(2)
        else:
            r = torch.zeros_like(ff)
        out = self.norm(r + ff)
        return out, mem, M, S, spk

    def forward_sequence(self, x_seq, mem, M, prev_spk, ablate_memory=False,
                         fpt_K=10):
        """FPT-parallel forward over the whole sequence.

        x_seq: (B, T, d_in)
        Returns: out_seq (B, T, d_mem), mem_final (B, N), M_final (B, dm, dm),
                 spk_seq (B, T, N) for return_stats use, final_spk (B, N).

        Pipeline:
          1. to_current (parallel over T)
          2. recurrent input: in FPT mode we use prev-spike from previous BPTT
             window only for t=0; for t>=1 we approximate using the spike at t-1
             from THIS sequence -- exact under FPT once converged.
          3. LIF: lif_parallel (FPT, K iterations of parallel scan)
          4. k/v/q/ff projections (parallel)
          5. Hebbian write/read (sequential loop -- light)
        """
        if self.delta_rule and self.learnable_decay:
            raise NotImplementedError(
                "chunked delta + per-dim learnable_decay not supported; use decay_gate "
                "(per-step scalar forget) instead, or sequential step() path.")
        if self.titans:
            raise NotImplementedError(
                "titans has no chunked kernel yet; train with use_fpt=false "
                "(sequential step_titans path).")
        from lif_parallel import lif_parallel
        B, T, _ = x_seq.shape
        cur_seq = self.to_current(x_seq)                       # (B, T, N)
        if self.recurrent:
            # prev_spk at t=0 from carry, at t>=1 from "previous-in-sequence".
            # Cheap approximation: use a one-step FPT preview. We use prev_spk
            # carry shifted in; the within-sequence recurrent contribution is
            # left to be picked up implicitly through the cur->spk->next-cur path
            # by re-running FPT (this is one of the K iterations).
            pad = prev_spk.unsqueeze(1)                        # (B, 1, N)
            # We don't know within-sequence prev_spk yet; use carry for t=0
            # and zeros for t>=1, then refine after spikes are known.
            zeros = torch.zeros(B, T - 1, self.n_neurons,
                                device=x_seq.device, dtype=x_seq.dtype)
            prev_seq_init = torch.cat([pad, zeros], dim=1)     # (B, T, N)
            cur_seq = cur_seq + F.linear(prev_seq_init,
                                          self.W_rec.weight * self.rec_mask)

        spk_seq, mem_final = lif_parallel(cur_seq, self.beta_val, vth=1.0,
                                          K=fpt_K)

        # If recurrent, refine once: now we know spk_seq, rebuild cur with the
        # correct prev_spk = shift_right(spk_seq, fill=carry), and re-run LIF.
        if self.recurrent:
            prev_seq = torch.cat([prev_spk.unsqueeze(1), spk_seq[:, :-1, :]],
                                 dim=1)
            cur_seq2 = self.to_current(x_seq) + F.linear(
                prev_seq, self.W_rec.weight * self.rec_mask)
            spk_seq, mem_final = lif_parallel(cur_seq2, self.beta_val, vth=1.0,
                                              K=fpt_K)

        # prev_spk shifted (k uses PREVIOUS token's spike)
        prev_seq = torch.cat([prev_spk.unsqueeze(1), spk_seq[:, :-1, :]], dim=1)
        # Pre-conv (depthwise causal 1D conv on spike sequence) before W_k/v/q.
        # FF residual uses unconvolved spk_seq (keep residual clean).
        if self.pre_conv:
            K = self.pre_conv_kernel
            spk_for_qv = F.pad(spk_seq.transpose(1, 2), (K - 1, 0))   # (B,N,T+K-1)
            spk_for_qv = self.spk_conv(spk_for_qv).transpose(1, 2)    # (B,T,N) causal
            prev_for_k = F.pad(prev_seq.transpose(1, 2), (K - 1, 0))
            prev_for_k = self.spk_conv(prev_for_k).transpose(1, 2)
        else:
            spk_for_qv = spk_seq
            prev_for_k = prev_seq
        k_seq = self.W_k(prev_for_k)                            # (B, T, dm)
        v_seq = self.W_v(spk_for_qv)
        q_seq = self.W_q(spk_for_qv)
        ff_seq = self.W_ff(spk_seq)

        # Hebbian: chunkwise-parallel via hebbian_chunked (TFLA-style).
        if self.write_gate:                                  # ST Phase 5
            v_seq = v_seq * torch.sigmoid(self.W_gate(spk_seq))
        if not ablate_memory:
            # Multi-head: split d_mem into H independent heads, collapse to batch.
            # (B,T,D) -> (B,T,H,Dh) -> (B*H, T, Dh) ; M is already (B*H, Dh, Dh).
            if self.n_heads > 1:
                H, Dh = self.n_heads, self.d_head
                v_in = v_seq.reshape(B, T, H, Dh).permute(0, 2, 1, 3).reshape(B * H, T, Dh)
                k_in = k_seq.reshape(B, T, H, Dh).permute(0, 2, 1, 3).reshape(B * H, T, Dh)
                q_in = q_seq.reshape(B, T, H, Dh).permute(0, 2, 1, 3).reshape(B * H, T, Dh)
            else:
                v_in, k_in, q_in = v_seq, k_seq, q_seq
            if self.delta_rule:                              # ST Phase 8 (chunked)
                if self.vector_beta:
                    # Per-channel β (RWKV-7 style). The chunked WY kernel assumes
                    # scalar β; use a sequential memory loop instead (LIF stays
                    # FPT-parallel, only the memory update is per-timestep).
                    # Slower (≈ titans cost) but correct; use small T/batch.
                    if self.n_heads > 1:
                        raise NotImplementedError("vector_beta + n_heads>1 not supported")
                    beta_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk_seq))  # (B,T,D)
                    alpha_seq = (torch.sigmoid(self.W_alpha(spk_seq)).clamp(0.5, 0.9999)
                                 if self.decay_gate else None)
                    r_list = []
                    for t in range(T):
                        kn = k_in[:, t] / (k_in[:, t].norm(dim=-1, keepdim=True) + 1e-6)
                        a = alpha_seq[:, t].unsqueeze(2) if alpha_seq is not None else self.lam
                        prevM = a * M
                        Mk = torch.bmm(prevM, kn.unsqueeze(2)).squeeze(2)
                        u = beta_seq[:, t] * (v_in[:, t] - Mk)            # (B,D)
                        M = prevM + torch.bmm(u.unsqueeze(2), kn.unsqueeze(1))
                        r_list.append(torch.bmm(M, q_in[:, t].unsqueeze(2)).squeeze(2))
                    r_in = torch.stack(r_list, dim=1)
                else:
                    beta_seq = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(self.W_beta(spk_seq))  # (B,T,1)
                    if self.n_heads > 1:
                        beta_in = beta_seq.unsqueeze(2).expand(B, T, self.n_heads, 1).reshape(B * self.n_heads, T, 1)
                    else:
                        beta_in = beta_seq
                    if self.decay_gate:
                        from delta_chunked import delta_chunked_gated
                        alpha_seq = torch.sigmoid(self.W_alpha(spk_seq)).clamp(0.5, 0.9999)
                        if self.n_heads > 1:
                            alpha_in = alpha_seq.unsqueeze(2).expand(B, T, self.n_heads, 1).reshape(B * self.n_heads, T, 1)
                        else:
                            alpha_in = alpha_seq
                        r_in, M = delta_chunked_gated(v_in, k_in, q_in, beta_in,
                                                       alpha_in, M, chunk=64)
                    else:
                        from delta_chunked import delta_chunked
                        r_in, M = delta_chunked(v_in, k_in, q_in, beta_in, M,
                                                 self.lam, chunk=64)
            elif self.learnable_decay:                       # ST Phase 4
                from hebbian_chunked import hebbian_gated_chunked
                alpha = torch.sigmoid(self.decay_raw).clamp(0.5, 0.9999)
                r_in, M = hebbian_gated_chunked(v_in, k_in, q_in, M,
                                                 alpha, self.eta, chunk=64)
            else:
                from hebbian_chunked import hebbian_chunked
                # chunk=256 (vs 64): ~4x fewer Python-launched kernels per layer,
                # which matters a lot uncompiled at long block_size (launch-bound).
                r_in, M = hebbian_chunked(v_in, k_in, q_in, M,
                                           self.lam, self.eta, chunk=256)
            # Multi-head: collapse heads back into d_mem dim.
            if self.n_heads > 1:
                H, Dh = self.n_heads, self.d_head
                r_seq = r_in.reshape(B, H, T, Dh).permute(0, 2, 1, 3).reshape(B, T, self.d_mem)
            else:
                r_seq = r_in
            if self.local_attn:
                # BASED-style sliding-window exact attention: scores = Q K^T /√D,
                # masked to (b<=t and t-b<W), softmax, then attn@V. Gated additive.
                D_ = q_seq.size(-1)
                scale = D_ ** -0.5
                scores = torch.bmm(q_seq, k_seq.transpose(1, 2)) * scale  # (B,T,T)
                idx = torch.arange(T, device=q_seq.device)
                diff = idx.unsqueeze(1) - idx.unsqueeze(0)
                m = (diff >= 0) & (diff < self.local_window)
                scores = scores.masked_fill(~m.unsqueeze(0), float('-inf'))
                attn = torch.softmax(scores, dim=-1)
                r_local = torch.bmm(attn, v_seq)
                r_seq = r_seq + torch.tanh(self.local_gate) * r_local
            out_seq = self.norm(r_seq + ff_seq)
        else:
            out_seq = self.norm(ff_seq)
        final_spk = spk_seq[:, -1, :]
        return out_seq, mem_final, M, spk_seq, final_spk


class SpikingHebbianLM(nn.Module):
    def __init__(self, vocab, d=128, n_neurons=256, d_mem=64,
                 beta=0.9, lam=0.98, eta=1.0, recurrent=False, rec_density=0.05,
                 compile_safe=False, tie_weights=False, n_layers=1,
                 use_fpt=False, fpt_K=10, learnable_decay=False, write_gate=False,
                 delta_rule=False, beta_floor=0.0, decay_gate=False,
                 titans=False, local_attn=False, local_window=64,
                 n_heads=1, pre_conv=False, pre_conv_kernel=4,
                 vector_beta=False):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.n_neurons = n_neurons
        self.d_mem = d_mem
        self.lam = lam
        self.eta = eta
        self.n_layers = n_layers
        self.compile_safe = compile_safe
        self.tie_weights = tie_weights
        self.use_fpt = use_fpt
        self.fpt_K = fpt_K

        if n_layers > 1 and d != d_mem:
            raise ValueError(f"n_layers > 1 requires d == d_mem for residual connections "
                             f"(got d={d}, d_mem={d_mem})")

        self.embed = nn.Embedding(vocab, d)
        blocks = []
        for i in range(n_layers):
            d_in = d if i == 0 else d_mem
            blocks.append(SpikingHebbianBlock(
                d_in, n_neurons, d_mem, beta, lam, eta,
                compile_safe, recurrent, rec_density,
                learnable_decay=learnable_decay, write_gate=write_gate,
                delta_rule=delta_rule, beta_floor=beta_floor,
                decay_gate=decay_gate, titans=titans,
                local_attn=local_attn, local_window=local_window,
                n_heads=n_heads,
                pre_conv=pre_conv, pre_conv_kernel=pre_conv_kernel,
                vector_beta=vector_beta))
        self.titans = titans
        self.n_heads = n_heads
        self.d_head = d_mem // n_heads
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d_mem, vocab)
        if tie_weights:
            if d != d_mem:
                raise ValueError(f"tie_weights requires d == d_mem (got d={d}, d_mem={d_mem})")
            self.head.weight = self.embed.weight

    def forward(self, idx, ablate_memory=False, return_stats=False,
                initial_state=None, return_final_state=False):
        B, T = idx.shape
        device = idx.device
        base = self.embed(idx)                                # (B, T, d)
        L = self.n_layers

        if initial_state is None:
            mems = [torch.zeros(B, self.n_neurons, device=device) for _ in range(L)]
            # n_heads>1: M is (B*H, d_head, d_head) (heads collapsed into batch dim)
            BM, DM = (B * self.n_heads, self.d_head) if self.n_heads > 1 else (B, self.d_mem)
            Ms = [torch.zeros(BM, DM, DM, device=device) for _ in range(L)]
            prev_spks = [torch.zeros(B, self.n_neurons, device=device) for _ in range(L)]
            Ss = [torch.zeros(BM, DM, DM, device=device) for _ in range(L)] if self.titans else [None] * L
        else:
            def _bcast(t, shape):
                if t.dim() == len(shape) - 1: t = t.unsqueeze(0)
                return t.to(device).expand(*shape).contiguous()
            layers = initial_state.get("layers", [initial_state])
            mems = [_bcast(layers[i]["mem"], (B, self.n_neurons)) for i in range(L)]
            Ms = [_bcast(layers[i]["M"], (B, self.d_mem, self.d_mem)) for i in range(L)]
            prev_spks = [_bcast(layers[i]["prev_spk"], (B, self.n_neurons)) for i in range(L)]
            if self.titans:
                Ss = [_bcast(layers[i].get("S", torch.zeros(self.d_mem, self.d_mem)),
                             (B, self.d_mem, self.d_mem)) for i in range(L)]
            else:
                Ss = [None] * L

        use_residual = (self.d == self.d_mem)

        if self.use_fpt:
            # FPT path: each block processes the FULL sequence in parallel.
            x_seq = base                                       # (B, T, d)
            spike_accum = []
            for l, block in enumerate(self.blocks):
                out_seq, mems[l], Ms[l], spk_seq, final_spk = block.forward_sequence(
                    x_seq, mems[l], Ms[l], prev_spks[l],
                    ablate_memory=ablate_memory, fpt_K=self.fpt_K)
                if L > 1 and (l > 0 or use_residual):
                    out_seq = out_seq + x_seq
                prev_spks[l] = final_spk
                x_seq = out_seq
                if return_stats:
                    spike_accum.append(spk_seq)
            logits = self.head(x_seq)                          # (B, T, vocab)
            all_spikes = spike_accum
        else:
            logits_list = []
            all_spikes = []
            for t in range(T):
                x = base[:, t, :]
                for l, block in enumerate(self.blocks):
                    if block.titans:
                        out, mems[l], Ms[l], Ss[l], spk = block.step_titans(
                            x, mems[l], Ms[l], Ss[l], prev_spks[l], ablate_memory)
                    else:
                        out, mems[l], Ms[l], spk = block.step(
                            x, mems[l], Ms[l], prev_spks[l], ablate_memory)
                    if L > 1 and (l > 0 or use_residual):
                        out = out + x
                    prev_spks[l] = spk
                    x = out
                logits_list.append(self.head(x))
                if return_stats:
                    all_spikes.extend(prev_spks)
            logits = torch.stack(logits_list, dim=1)

        spike_rate = None
        if return_stats:
            if self.use_fpt:
                spike_rate = torch.stack([s.mean() for s in all_spikes]).mean()
            else:
                spike_rate = torch.stack(all_spikes).mean()
        if return_final_state:
            state = {"layers": [
                {"M": Ms[i].detach(), "mem": mems[i].detach(), "prev_spk": prev_spks[i].detach()}
                for i in range(L)
            ]}
            # Stateful training needs both the spike rate (for the sparsity
            # penalty) and the carried state in a single forward pass.
            if return_stats:
                return logits, spike_rate, state
            return logits, state
        if return_stats:
            return logits, spike_rate
        return logits

    def save_brain(self, state, path):
        """Persist a complete neuron state ('brain') to disk."""
        def _to1(t):
            if t.dim() >= 2 and t.shape[0] == 1: return t.squeeze(0)
            return t
        layers = state.get("layers", [state])
        blob = {
            "layers": [{
                "M": _to1(lay["M"]).detach().cpu(),
                "mem": _to1(lay["mem"]).detach().cpu(),
                "prev_spk": _to1(lay["prev_spk"]).detach().cpu(),
            } for lay in layers],
            "n_layers": len(layers),
            "d_mem": self.d_mem,
            "n_neurons": self.n_neurons,
            "vocab": self.vocab,
            "lam": self.lam,
            "eta": self.eta,
            "format_version": 3,
        }
        torch.save(blob, path)

    @staticmethod
    def load_brain(path):
        """Load a brain file. Returns a state dict suitable for initial_state=."""
        blob = torch.load(path, weights_only=False)
        if blob.get("format_version", 1) < 3:
            return {"layers": [{"M": blob["M"], "mem": blob["mem"], "prev_spk": blob["prev_spk"]}]}
        return {"layers": blob["layers"]}
