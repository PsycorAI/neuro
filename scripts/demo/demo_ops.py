"""demo_ops.py -- 2026-10-25 (Neuro 78, decode-step kernel fusion, Glenn: "fuse the tiny kernels"). The decoder step's two
hand-written Triton kernels registered as torch custom ops, so torch.compile can fuse everything AROUND them in one graph
instead of breaking the step into ~28 pieces at each call:
  n78::lif_step(cur, beta_t, beta_f, ml, gamma, levels, signed, alpha) -> (spk, mem)   the exact LIF (src/lif_exact.py);
      multi-level (ml) takes the per-neuron beta tensor, the binary kernel a python-float beta (no host sync in a graph)
  n78::sparse_linear(x, wt, bias) -> y                                                the W3 sparse GEMV (sparse_gemv.py)
Eager calls behave exactly as calling the kernels directly (outputs are fresh tensors: no aliasing)."""
from typing import Optional, Tuple
import torch
from torch import Tensor


@torch.library.custom_op("n78::lif_step", mutates_args=())
def lif_step(cur: Tensor, beta_t: Tensor, beta_f: float, ml: bool, gamma: float, levels: int, signed: bool,
             alpha: float) -> Tuple[Tensor, Tensor]:
    from lif_exact import lif_exact, lif_exact_ml
    if ml:
        return lif_exact_ml(cur, beta_t, vth=1.0, alpha=alpha, gamma=gamma, levels=levels, signed=signed)
    return lif_exact(cur, beta_f, vth=1.0, alpha=alpha, gamma=gamma)


@lif_step.register_fake
def _(cur, beta_t, beta_f, ml, gamma, levels, signed, alpha):
    return torch.empty_like(cur), cur.new_empty((cur.shape[0], cur.shape[2]))


@torch.library.custom_op("n78::sparse_linear", mutates_args=())
def sparse_linear(x: Tensor, wt: Tensor, bias: Optional[Tensor]) -> Tensor:
    from sparse_gemv import sparse_linear_1tok
    return sparse_linear_1tok(x.contiguous(), wt, bias)


@sparse_linear.register_fake
def _(x, wt, bias):
    return x.new_empty((wt.shape[1],))
