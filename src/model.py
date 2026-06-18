"""Phase 1: Spiking Hebbian language model (the core).

Token model where:
  - a Leaky integrate-and-fire (LIF) layer emits sparse binary spikes (the "neurons")
  - a decaying Hebbian fast-weight matrix M stores outer-product associations (the "synapses")
  - M binds the PREVIOUS token's key to the CURRENT token's value, so a later
    occurrence of a token can retrieve its learned successor (induction / recall).

      M_t = lambda * M_{t-1} + eta * (v_t  outer  k_{t-1})
      r_t = M_t q_t

This is the BDH "working memory lives on the synapses" claim, actually implemented:
the memory is a state written and read at inference time, not a static weight. The
memory cost is O(d_mem^2) per token and INDEPENDENT of context length.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import Leaky, surrogate


class SpikingHebbianLM(nn.Module):
    def __init__(self, vocab, d=128, n_neurons=256, d_mem=64,
                 beta=0.9, lam=0.98, eta=1.0, recurrent=False, rec_density=0.05,
                 compile_safe=False, tie_weights=False):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.n_neurons = n_neurons
        self.d_mem = d_mem
        self.lam = lam
        self.eta = eta
        self.recurrent = recurrent
        self.compile_safe = compile_safe
        self.tie_weights = tie_weights
        self.beta_val = beta

        self.embed = nn.Embedding(vocab, d)
        self.to_current = nn.Linear(d, n_neurons)
        if not compile_safe:
            self.lif = Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        if recurrent:
            self.W_rec = nn.Linear(n_neurons, n_neurons, bias=False)   # neuron->neuron synapses
            mask = (torch.rand(n_neurons, n_neurons) < rec_density).float()
            mask.fill_diagonal_(0)
            self.register_buffer("rec_mask", mask)
        self.W_k = nn.Linear(n_neurons, d_mem, bias=False)   # key   <- previous spikes
        self.W_v = nn.Linear(n_neurons, d_mem, bias=False)   # value <- current spikes
        self.W_q = nn.Linear(n_neurons, d_mem, bias=False)   # query <- current spikes
        self.W_ff = nn.Linear(n_neurons, d_mem, bias=False)  # static feed-forward path
        self.norm = nn.LayerNorm(d_mem)
        self.head = nn.Linear(d_mem, vocab)
        if tie_weights:
            if d != d_mem:
                raise ValueError(f"tie_weights requires d == d_mem (got d={d}, d_mem={d_mem})")
            # share the (vocab, d) weight matrix between input embedding and output head.
            # Saves vocab*d params (typically ~40% on small models) and often improves
            # perplexity at small scale by enforcing input/output representation symmetry.
            self.head.weight = self.embed.weight

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

    def forward(self, idx, ablate_memory=False, return_stats=False,
                initial_state=None, return_final_state=False):
        """initial_state: optional dict {M, mem, prev_spk} to resume from.
        return_final_state: if True, also return the final neuron state dict."""
        B, T = idx.shape
        device = idx.device
        base = self.to_current(self.embed(idx))              # (B,T,N)

        if initial_state is None:
            mem = torch.zeros(B, self.n_neurons, device=device)
            M = torch.zeros(B, self.d_mem, self.d_mem, device=device)
            prev_spk = torch.zeros(B, self.n_neurons, device=device)
        else:
            def _bcast(t, shape):
                if t.dim() == len(shape) - 1: t = t.unsqueeze(0)
                return t.to(device).expand(*shape).contiguous()
            mem = _bcast(initial_state["mem"], (B, self.n_neurons))
            M = _bcast(initial_state["M"], (B, self.d_mem, self.d_mem))
            prev_spk = _bcast(initial_state["prev_spk"], (B, self.n_neurons))

        logits, spikes = [], []
        for t in range(T):
            inp = base[:, t, :]
            if self.recurrent:                               # neuron->neuron synaptic input
                inp = inp + F.linear(prev_spk, self.W_rec.weight * self.rec_mask)
            if self.compile_safe:
                spk, mem = self._lif_step(inp, mem)
            else:
                spk, mem = self.lif(inp, mem)                # (B,N) spikes in {0,1}
            spikes.append(spk)
            k = self.W_k(prev_spk)
            v = self.W_v(spk)
            q = self.W_q(spk)
            ff = self.W_ff(spk)                              # static path: local statistics
            if not ablate_memory:
                # Hebbian write: bind previous token -> current token, with decay
                M = self.lam * M + self.eta * torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
                r = torch.bmm(M, q.unsqueeze(2)).squeeze(2)  # associative read (in-context)
            else:
                r = torch.zeros(B, self.d_mem, device=device)
            logits.append(self.head(self.norm(r + ff)))
            prev_spk = spk

        logits = torch.stack(logits, dim=1)                  # (B,T,vocab)
        if return_final_state:
            state = {"M": M.detach(), "mem": mem.detach(), "prev_spk": prev_spk.detach()}
            return logits, state
        if return_stats:
            spike_rate = torch.stack(spikes).mean()          # differentiable scalar
            return logits, spike_rate
        return logits

    def save_brain(self, state, path):
        """Persist a complete neuron state ('brain') to disk."""
        def _to1(t):
            if t.dim() == 3 and t.shape[0] == 1: return t.squeeze(0)
            if t.dim() == 2 and t.shape[0] == 1: return t.squeeze(0)
            return t
        torch.save({
            "M": _to1(state["M"]).detach().cpu(),
            "mem": _to1(state["mem"]).detach().cpu(),
            "prev_spk": _to1(state["prev_spk"]).detach().cpu(),
            "d_mem": self.d_mem,
            "n_neurons": self.n_neurons,
            "vocab": self.vocab,
            "lam": self.lam,
            "eta": self.eta,
            "format_version": 2,
        }, path)

    @staticmethod
    def load_brain(path):
        """Load a brain file. Returns a state dict suitable for initial_state=."""
        blob = torch.load(path, weights_only=False)
        return {"M": blob["M"], "mem": blob["mem"], "prev_spk": blob["prev_spk"]}
