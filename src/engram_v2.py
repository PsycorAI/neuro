"""Engram v2: residual-stream conditional memory (DeepSeek-V4-aligned redesign).

Why a v2 (see idea C in docs/to-do.html and memory neuro_engram)
----------------------------------------------------------------
The original `engram.py` closed inconclusive-to-negative at 30M. Research on
DeepSeek V4's shipped Engram (arXiv/introl, Jan-Apr 2026) pinned two design
flaws in ours, not a bad idea:

  1. WRONG INJECTION POINT. v1 added a vocab LOGIT BIAS at the output
     (`logits += engram(idx)`), so retrieved knowledge could only nudge the
     final token distribution -- it never entered the network's reasoning.
     DeepSeek injects a retrieved VALUE VECTOR into the RESIDUAL STREAM at an
     early/mid layer (before the block's mixing), so the fact participates in
     all downstream computation. This module does that: idx -> (B,T,d) vector
     added into the hidden state.

  2. WRONG METRIC. v1 was gated on general val_ppl, which barely moves with
     factual memory (DeepSeek's own ablation: general capability ~flat, but
     factual benchmarks collapse to 29-44% without Engram). That harness lives
     separately (scripts/engram_factual_eval.py) -- this module is only the
     mechanism.

Also folds in the two cheaper fixes from the research:
  3. MULTI-HEAD HASHING: H independent hashes per n-gram order; a collision in
     one head is corrected by the others (collisions corrupt facts).
  4. MULTI-GRANULAR N-GRAMS: mix orders (e.g. N=2,3) so both short and longer
     factual contexts are addressable.

Relationship to the rest of Neuro
---------------------------------
This is the STATIC long-term store; the delta_rule/Hebbian fast-weights are the
DYNAMIC working memory. Two complementary systems (the split DeepSeek frames as
"what you know" vs "how you think"), matching reference_delta_hebbian_combine.

Cost / safety
-------------
Lookup is O(1): a few hashes + embedding gathers, no dense matmul on the memory
path. The output projection is ZERO-initialized, so at init the module adds
exactly nothing to the residual -- safe to drop into a trained checkpoint and
fine-tune, or to train from scratch, without perturbing early dynamics.

NOTE: standalone module. Wiring into model.py (add `emb = emb + engram_v2(idx)`
before the block stack, behind an `engram_v2_enabled` config flag) is a separate
one-line dispatch, deliberately not done here while live runs could restart
(see memory feedback_live_run_code_safety).
"""
import torch
import torch.nn as nn


# Odd primes coprime with 2^k, for bit-mixing across the token vocab. Need
# >= max_n * n_heads distinct multipliers so every (order, head) hash differs.
_PRIMES = (
    2654435761, 2246822519, 3266489917, 668265263, 374761393,
    3862899647, 1367130551, 2809346533, 1908281009, 4184922817,
    3467471779, 2596859587, 1521134513, 3963737609, 2135587861,
    1264345823, 3011652109, 4093492433, 1811939329, 2543568647,
    # Extended 2026-07-16 to unblock engram_v2_heads >= 7 (was capped
    # at heads=6 by original 20-entry list, orders=[2,3]):
    2147483647, 998244353,  469762049,  754974721,  167772161,
    2013265921, 2281701377, 3489660929, 3892314113, 985661441,
    1224736769, 3221225473,
)


class EngramV2(nn.Module):
    """Residual-stream conditional memory.

    Args:
        d:      model hidden dim (output vectors are (B, T, d), added to residual)
        vocab:  input token vocabulary size (only for hash range sanity; unused otherwise)
        orders: tuple of n-gram orders to hash, e.g. (2, 3) for bi+tri-gram
        K:      hash buckets per (order, head), MUST be power of 2
        heads:  independent hashes per order (collision-robustness)
        rank:   low-rank dim of each bucket value before projecting to d

    Forward:
        idx: (B, T) LongTensor of input token IDs
        returns: (B, T, d) FloatTensor to ADD into the residual stream
    """

    def __init__(self, d, vocab, orders=(2, 3), K=65536, heads=2, rank=64, hash_hi=False):
        super().__init__()
        # hash_hi (2026-10-02, r16 b1, docs/code_review_interpretability_2026_10_01.md #1; default False =
        # byte-identical): the low-bit hash keeps (sum p_i*id_i) & (K-1), whose low log2(K) bits depend only
        # on the low bits of each id -> ids x and x+-K share a bucket in EVERY table. hash_hi takes the TOP
        # log2(K) bits of the 32-bit product sum instead (Knuth multiplicative hashing).
        self.hash_hi = bool(hash_hi)
        self.shift = 32 - (K.bit_length() - 1)
        assert K > 0 and (K & (K - 1)) == 0, f"K must be power of 2, got {K}"
        max_n = max(orders)
        # Prime consumption is len(orders) * heads * max_n (see hash-fill loop
        # below); prior assert used max_n*heads which undercounted by len(orders)
        # and let e.g. heads=8/orders=(2,3) silently wrap into reused multipliers
        # across tables. Fixed 2026-07-16.
        need = len(orders) * heads * max_n
        assert len(_PRIMES) >= need, f"need >= {need} primes, have {len(_PRIMES)}"
        self.d = d
        self.orders = tuple(orders)
        self.K = K
        self.heads = heads
        self.rank = rank
        self.mask = K - 1
        # One bucket table per (order, head). Concatenated retrieved ranks are
        # projected to d by a single shared output layer.
        self.n_tables = len(self.orders) * heads
        self.bucket = nn.Embedding(K * self.n_tables, rank)
        nn.init.normal_(self.bucket.weight, std=0.001)   # tiny, nonzero: proj gets grad from step 1
        self.proj = nn.Linear(rank * self.n_tables, d, bias=False)
        nn.init.zeros_(self.proj.weight)                 # ZERO: no-op at init, safe residual add
        # Per-(order,head) prime multipliers, laid out [table, order_pos].
        primes = torch.zeros(self.n_tables, max_n, dtype=torch.long)
        pi = 0
        table = 0
        for _ in self.orders:
            for _h in range(heads):
                for pos in range(max_n):
                    primes[table, pos] = _PRIMES[pi % len(_PRIMES)]
                    pi += 1
                table += 1
        self.register_buffer("primes", primes, persistent=False)
        # Table-base offsets so each (order,head) indexes a disjoint K-slice of
        # the shared Embedding.
        self.register_buffer(
            "table_base",
            torch.arange(self.n_tables, dtype=torch.long) * K,
            persistent=False,
        )

    def _hash_order_head(self, idx, n, table):
        """Bucket index (B, T) for one n-gram order and one hash head/table.

        h = sum_{i=0..n-1} primes[table, i] * idx[t-(n-1)+i]  (mod 2^32) & (K-1)
        Left-pad with 0 for positions with < n-1 left context.
        """
        B, T = idx.shape
        pad = torch.zeros(B, n - 1, dtype=idx.dtype, device=idx.device)
        idx_p = torch.cat([pad, idx], dim=1)               # (B, T+n-1)
        h = torch.zeros(B, T, dtype=torch.long, device=idx.device)
        for i in range(n):
            h = h + self.primes[table, i] * idx_p[:, i:i + T].long()
        h = h & 0xFFFFFFFF
        h = (h >> self.shift) if self.hash_hi else (h & self.mask)
        return h + self.table_base[table]                  # into shared table's slice

    def forward(self, idx):
        """(B, T) idx -> (B, T, d) residual increment."""
        B, T = idx.shape
        parts = []
        table = 0
        for n in self.orders:
            for _h in range(self.heads):
                h = self._hash_order_head(idx, n, table)   # (B, T)
                parts.append(self.bucket(h))               # (B, T, rank)
                table += 1
        cat = torch.cat(parts, dim=-1)                     # (B, T, rank * n_tables)
        return self.proj(cat)                              # (B, T, d)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, d, V = 4, 32, 256, 16384
    m = EngramV2(d=d, vocab=V, orders=(2, 3), K=1024, heads=2, rank=64)
    idx = torch.randint(0, V, (B, T))
    out = m(idx)
    assert out.shape == (B, T, d), out.shape
    # Zero-init proj -> exact no-op at init (safe residual add).
    assert torch.count_nonzero(out) == 0, "output must be exactly zero at init"
    # After a gradient step the module becomes active and stays finite.
    tgt = torch.randn(B, T, d)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    loss = ((m(idx) - tgt) ** 2).mean()
    loss.backward()
    opt.step()
    out2 = m(idx)
    assert torch.count_nonzero(out2) > 0, "module should be active after a step"
    assert torch.isfinite(out2).all()
    # Collision-robustness sanity: two different contexts should (almost always)
    # produce different retrievals across the multi-head/multi-order tables.
    a = m(torch.zeros(1, 3, dtype=torch.long))
    b = m(torch.arange(3).view(1, 3))
    assert not torch.allclose(a, b), "distinct contexts collided across all tables"
    print(f"[OK] EngramV2: shape {tuple(out.shape)}, zero-init no-op, "
          f"active+finite after step, {m.n_tables} tables, "
          f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M params")
