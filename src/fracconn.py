"""Frac-Connections (arXiv:2503.14125): a learnable generalization of the
plain residual add, at near-zero extra cost.

Why (see docs/hyperconnections_design.md for the full design note)
--------------------------------------------------------------
Full Hyper-Connections (ByteDance, arXiv:2409.19606) widen the residual
stream n x, which we rejected on the same 16GB-headroom grounds that killed
`vector_beta`. Frac-Connections instead SPLIT the existing d-dim stream into
n fractions of width d/n and learn how those fractions mix across the block
boundary -- same memory as a plain residual, just a handful of extra scalars
per layer.

The block itself never changes: it still receives and returns a full d-dim
tensor (`forward_sequence`, the delta rule, the FLA kernel, MoR router,
sparsity, beta-floor all stay byte-identical). Only the inter-block "how do
input and output combine into the next input" step changes, replacing the
fixed `x_new = out + x` with a learned per-layer mix over the n fractions.

Safety: EXACT-IDENTITY INIT. Both mixing matrices (W for the block's own
output, V for the carried-over input) start at the identity matrix, so at
step 0 `FracConnection(out, x) == out + x` exactly (bit-for-bit, verified by
`temp/check_fracconn_identity.py`) -- restart-safe to drop into any existing
checkpoint or fresh run without perturbing a regime we've already tuned.

NOTE: standalone module. Wiring into model.py (replacing the 3 mirrored
`out_seq = out_seq + x_seq` sites with a call to this module, behind a
`fracconn_enabled` config flag) is the model.py change; kept separate so the
identity-init guarantee can be unit-tested in isolation first.
"""
import torch
import torch.nn as nn


class FracConnection(nn.Module):
    """One layer's worth of Frac-Connection mixing.

    Args:
        d:      stream width (must be divisible by n_frac)
        n_frac: number of fractions to split the stream into (paper default: 2)

    Forward:
        out, x: (..., d) -- the block's output and the carried-over input
        returns: (..., d) -- the mixed stream to feed to the next layer,
                 == out + x at init (W = V = identity)
    """

    def __init__(self, d, n_frac=2):
        super().__init__()
        assert d % n_frac == 0, f"d={d} not divisible by n_frac={n_frac}"
        self.n_frac = n_frac
        self.d_frac = d // n_frac
        # W mixes the block's OUTPUT fractions into each new fraction; V mixes
        # the carried INPUT fractions. Identity init -> new_frac[i] =
        # out_frac[i] + x_frac[i] for all i, i.e. exactly the plain residual.
        self.W = nn.Parameter(torch.eye(n_frac))
        self.V = nn.Parameter(torch.eye(n_frac))

    def forward(self, out, x):
        out_f = out.split(self.d_frac, dim=-1)
        x_f = x.split(self.d_frac, dim=-1)
        new_f = []
        for i in range(self.n_frac):
            acc = self.W[i, 0] * out_f[0] + self.V[i, 0] * x_f[0]
            for j in range(1, self.n_frac):
                acc = acc + self.W[i, j] * out_f[j] + self.V[i, j] * x_f[j]
            new_f.append(acc)
        return torch.cat(new_f, dim=-1)
