"""spike_code.py -- 2026-10-06 (energy plan, docs/energy_plan_2026_10_06.md; default off).
Signed multi-level integer spike coding of a linear layer's INPUT (SpikingBrain integer spike counts / SpikingMamba
TI-LIF style): s = clamp(round(x / theta), -L, L) (0..L if unsigned), output s * theta. theta is per input channel and
learned (LSQ-style: the straight-through round makes d out / d theta = round(u) - u inside the range, +-L when clipped).
On neuromorphic hardware W @ (s * theta) = (W * theta) @ s, i.e. |s| accumulates (ACs) per nonzero input instead of a
dense multiply-accumulate (MAC) per input; zeros cost nothing. theta starts at `init` (the RMS-normed inputs have
mean|x| ~0.8); conversion runs set it from real activations first (calibrate()). Stats (no grad, in-place buffers ->
compile-safe): last_count = mean |s| (spikes per input element), last_nz = fraction of nonzero s."""
import torch
import torch.nn as nn


class SpikeCode(nn.Module):
    def __init__(self, dim, levels=4, signed=True, init=1.0):
        super().__init__()
        self.L, self.signed = int(levels), bool(signed)
        self.theta = nn.Parameter(torch.full((dim,), float(init)))
        self.register_buffer("last_count", torch.zeros(()), persistent=False)
        self.register_buffer("last_nz", torch.zeros(()), persistent=False)
        self.calib = None                     # set by calibrate(): per-channel mean|x| accumulator

    def forward(self, x):
        if self.calib is not None:            # calibration pass: record mean|x|, pass x through uncoded
            with torch.no_grad():
                a = x.detach().float().abs().reshape(-1, x.shape[-1])
                if not self.signed:           # sparse nonnegative input (relu2 hidden): mean over NONZERO entries
                    a = a.sum(0) / (a > 0).float().sum(0).clamp(min=1)
                    self.calib.append(a)
                else:
                    self.calib.append(a.mean(0))
            return x
        th = self.theta.abs().to(x.dtype) + 1e-4
        u = x / th
        lo = -self.L if self.signed else 0
        s = u.clamp(lo, self.L)
        s = s + (s.round() - s).detach()      # straight-through round
        with torch.no_grad():
            a = s.detach().abs()
            self.last_count.copy_(a.float().mean())
            self.last_nz.copy_((a > 0).float().mean())
        return s * th


@torch.no_grad()
def calibrate(model, batches, scale=1.0):
    """theta = scale * mean|x| per channel, measured on the given input batches with coding bypassed."""
    codes = [m for m in model.modules() if isinstance(m, SpikeCode)]
    for m in codes:
        m.calib = []
    was = model.training
    model.eval()
    for x in batches:
        model(x)
    for m in codes:
        m.theta.copy_((scale * torch.stack(m.calib).mean(0)).clamp(min=1e-3))
        m.calib = None
    model.train(was)
    return len(codes)
