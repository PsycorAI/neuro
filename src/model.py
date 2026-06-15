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
import torch
import torch.nn as nn
from snntorch import Leaky, surrogate


class SpikingHebbianLM(nn.Module):
    def __init__(self, vocab, d=128, n_neurons=256, d_mem=64,
                 beta=0.9, lam=0.98, eta=1.0):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.n_neurons = n_neurons
        self.d_mem = d_mem
        self.lam = lam
        self.eta = eta

        self.embed = nn.Embedding(vocab, d)
        self.to_current = nn.Linear(d, n_neurons)
        self.lif = Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.W_k = nn.Linear(n_neurons, d_mem, bias=False)   # key   <- previous spikes
        self.W_v = nn.Linear(n_neurons, d_mem, bias=False)   # value <- current spikes
        self.W_q = nn.Linear(n_neurons, d_mem, bias=False)   # query <- current spikes
        self.W_ff = nn.Linear(n_neurons, d_mem, bias=False)  # static feed-forward path
        self.norm = nn.LayerNorm(d_mem)
        self.head = nn.Linear(d_mem, vocab)

    def forward(self, idx, ablate_memory=False, return_stats=False):
        B, T = idx.shape
        device = idx.device
        cur = self.to_current(self.embed(idx))               # (B,T,N)

        mem = torch.zeros(B, self.n_neurons, device=device)
        M = torch.zeros(B, self.d_mem, self.d_mem, device=device)   # synaptic memory
        prev_spk = torch.zeros(B, self.n_neurons, device=device)

        logits, spikes = [], []
        for t in range(T):
            spk, mem = self.lif(cur[:, t, :], mem)           # (B,N) spikes in {0,1}
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
        if return_stats:
            spike_rate = torch.stack(spikes).mean()          # differentiable scalar
            return logits, spike_rate
        return logits
