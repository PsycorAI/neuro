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
                 compile_safe=False, recurrent=False, rec_density=0.05):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_mem = d_mem
        self.lam = lam
        self.eta = eta
        self.compile_safe = compile_safe
        self.beta_val = beta

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
        if not ablate_memory:
            M = self.lam * M + self.eta * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
            r = torch.bmm(M, q.unsqueeze(2)).squeeze(2)
        else:
            r = torch.zeros_like(ff)
        out = self.norm(r + ff)
        return out, mem, M, spk

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
        k_seq = self.W_k(prev_seq)                             # (B, T, dm)
        v_seq = self.W_v(spk_seq)
        q_seq = self.W_q(spk_seq)
        ff_seq = self.W_ff(spk_seq)

        # Hebbian: sequential but LIGHT (just two bmm per step).
        # M[t] = lam * M[t-1] + eta * v[t] outer k[t]
        # r[t] = M[t] @ q[t]
        outs = []
        if not ablate_memory:
            for t in range(T):
                M = self.lam * M + self.eta * torch.bmm(
                    v_seq[:, t, :].unsqueeze(2), k_seq[:, t, :].unsqueeze(1))
                r = torch.bmm(M, q_seq[:, t, :].unsqueeze(2)).squeeze(2)
                outs.append(self.norm(r + ff_seq[:, t, :]))
        else:
            for t in range(T):
                outs.append(self.norm(ff_seq[:, t, :]))
        out_seq = torch.stack(outs, dim=1)
        final_spk = spk_seq[:, -1, :]
        return out_seq, mem_final, M, spk_seq, final_spk


class SpikingHebbianLM(nn.Module):
    def __init__(self, vocab, d=128, n_neurons=256, d_mem=64,
                 beta=0.9, lam=0.98, eta=1.0, recurrent=False, rec_density=0.05,
                 compile_safe=False, tie_weights=False, n_layers=1,
                 use_fpt=False, fpt_K=10):
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
                compile_safe, recurrent, rec_density))
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
            Ms = [torch.zeros(B, self.d_mem, self.d_mem, device=device) for _ in range(L)]
            prev_spks = [torch.zeros(B, self.n_neurons, device=device) for _ in range(L)]
        else:
            def _bcast(t, shape):
                if t.dim() == len(shape) - 1: t = t.unsqueeze(0)
                return t.to(device).expand(*shape).contiguous()
            layers = initial_state.get("layers", [initial_state])
            mems = [_bcast(layers[i]["mem"], (B, self.n_neurons)) for i in range(L)]
            Ms = [_bcast(layers[i]["M"], (B, self.d_mem, self.d_mem)) for i in range(L)]
            prev_spks = [_bcast(layers[i]["prev_spk"], (B, self.n_neurons)) for i in range(L)]

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

        if return_final_state:
            state = {"layers": [
                {"M": Ms[i].detach(), "mem": mems[i].detach(), "prev_spk": prev_spks[i].detach()}
                for i in range(L)
            ]}
            return logits, state
        if return_stats:
            if self.use_fpt:
                spike_rate = torch.stack([s.mean() for s in all_spikes]).mean()
            else:
                spike_rate = torch.stack(all_spikes).mean()
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
