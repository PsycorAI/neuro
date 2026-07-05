"""Engram: n-gram hashed factual-recall store, DeepSeek-style.

Concept
-------
A learned lookup table indexed by a multiplicative hash of the last N input
tokens. Complements the dynamic Hebbian fast-weight memory (Ms) with a STATIC,
long-term factual prior that costs no matmul at inference -- just a hash
compute and two embedding lookups.

Why it fits Neuro's goal set
----------------------------
- Faster / cheaper inference: hash + lookup replaces a dense matmul for the
  fraction of prediction that's driven by memorized n-gram statistics.
- Lower energy: no dense compute for the memory path.
- Auditable: each bucket is literally inspectable -- you can print the top
  vocab biases for any n-gram context and see the memorized association.
- Dual-memory story: static Engram + dynamic Hebbian = "two memory systems
  like the brain" narrative, symmetric with the existing MTP/confidence/markov
  auxiliary heads architecture.

Position in the model
---------------------
Output-side logit bias (parallel to the Batch C Markov head, but higher-order):
    logits = head(x_seq) + markov_correction + engram(idx)
Each is an input-conditioned bias:
    markov: unigram/low-rank projection of the CURRENT input token
    engram: n-gram hash of the last N tokens -> low-rank -> vocab bias

Sizing (at 30M scale, d=1024, vocab=16384)
-----------------------------------------
Params = K * rank + rank * vocab
With K=32768, rank=64: 2.1M + 1.0M = 3.1M -- parity with markov head budget.

Hashing
-------
Multiplicative rolling hash over the last N token IDs, GPU-vectorized. Odd
prime multipliers give good bucket distribution across a token vocab. Buckets
are power-of-two (K = 2^k) so index reduction is a bitmask, not a mod.

For positions t < N-1 (not enough left context yet), we left-pad idx with 0
(typically BOS/pad in this tokenizer).
"""
import torch
import torch.nn as nn


# Odd primes, coprime with 2^k. Chosen for good bit-mixing across a 16k vocab.
_ENGRAM_PRIMES = (2654435761, 2246822519, 3266489917, 668265263, 374761393)


class EngramLookup(nn.Module):
    """n-gram hash -> low-rank bucket vector -> vocab bias.

    Args:
        vocab: output vocabulary size (bias is added to logits of shape (..., vocab))
        n:     n-gram order (n=3 = trigram over last 3 tokens)
        K:     number of hash buckets (MUST be a power of 2)
        rank:  low-rank dim of each bucket vector

    Forward:
        idx: (B, T) LongTensor of input token IDs
        returns: (B, T, vocab) FloatTensor to be added to output logits

    The bucket table is zero-initialized so the module contributes nothing at
    init (safe start); the projection is small-normal-init so once buckets
    learn nonzero values, the mapping to vocab is well-scaled.
    """

    def __init__(self, vocab, n=3, K=32768, rank=64):
        super().__init__()
        assert n >= 1
        assert K > 0 and (K & (K - 1)) == 0, f"K must be power of 2, got {K}"
        assert len(_ENGRAM_PRIMES) >= n, f"need >= {n} primes, have {len(_ENGRAM_PRIMES)}"
        self.vocab = vocab
        self.n = n
        self.K = K
        self.rank = rank
        self.mask = K - 1
        # Multipliers as buffer so they move with the module (device) but
        # aren't parameters.
        self.register_buffer(
            "primes",
            torch.tensor(_ENGRAM_PRIMES[:n], dtype=torch.long),
            persistent=False,
        )
        self.bucket = nn.Embedding(K, rank)
        # Tiny (not zero) init: bias magnitude at start ~ 1e-4 at logit scale
        # (safe, doesn't perturb early training) but nonzero, so proj gets
        # gradient signal from step 1 rather than waiting on buckets to
        # accumulate first. Zero-init both sides would slow-bootstrap the proj.
        nn.init.normal_(self.bucket.weight, std=0.001)
        self.proj = nn.Linear(rank, vocab, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def _hash(self, idx):
        """Compute the bucket index for every (B, T) position.

        h = sum_{i=0..n-1} primes[i] * idx[t - (n-1) + i]   (mod 2^32)
        then AND with (K-1) to fold into K buckets. Left-pad with 0 for
        positions where t < n-1 so the shape stays (B, T).
        """
        B, T = idx.shape
        n = self.n
        # Left-pad with 0 so pad[:, k] indexes token at position (t - (n-1) + k)
        pad = torch.zeros(B, n - 1, dtype=idx.dtype, device=idx.device)
        idx_padded = torch.cat([pad, idx], dim=1)          # (B, T + n - 1)
        h = torch.zeros(B, T, dtype=torch.long, device=idx.device)
        for k in range(n):
            h = h + self.primes[k].to(idx.device) * idx_padded[:, k:k + T].long()
        # Fold into 32 bits then mask to K buckets. 64-bit longs stay signed,
        # so mask with (2^32 - 1) first to strip sign extension before final AND.
        h = h & 0xFFFFFFFF
        h = h & self.mask
        return h                                            # (B, T)

    def forward(self, idx):
        """(B, T) idx -> (B, T, vocab) logit bias."""
        h = self._hash(idx)
        r = self.bucket(h)          # (B, T, rank)
        bias = self.proj(r)         # (B, T, vocab)
        return bias
