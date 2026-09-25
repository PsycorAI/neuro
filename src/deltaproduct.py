"""DeltaProduct (arXiv:2502.10297) for the GDN-2 memory path.

Each token undergoes n_h Householder-product updates instead of one, lifting the
state-transition from rank-1 (DeltaNet) toward rank-n_h -- more gradient-descent
steps per token on the associative-recall loss, the axis where the spiking model
trails the transformer at high N.

Implementation: NO new kernel. We expand the sequence T -> T*n_h (n_h sub-steps
per token, each with its own k/v/erase/write/decay projections) and reuse the
existing gdn2_chunked_dense recurrence, then read out the state after each
token last sub-step. The equivalence of this expansion to the explicit
Householder-product form is verified in temp/deltaproduct_reference.py
(max abs err 3.5e-15; n_h=1 reduces to DeltaNet exactly).

Sub-step 1 reuses the block existing W_k/W_v/W_b/W_w/W_fdecay/gdn2_a so that
n_h=1 is byte-identical to the current model (this module is never constructed
then). Sub-steps 2..n_h get their own projections here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaProductExtra(nn.Module):
    """Projections + gates for DeltaProduct sub-steps 2..n_h.

    Mirrors the GDN-2 gate machinery (erase=sigmoid, write=floored-sigmoid,
    decay=exp(-exp(a)*softplus)) with the same init as the block main gates,
    so every sub-step is a well-formed GDN-2 update.
    """
    def __init__(self, n_neurons, d_mem, nh, beta_floor, gate_norm,
                 norm_ctor=None):
        super().__init__()
        assert nh >= 2
        self.nh = nh
        self.beta_floor = beta_floor
        self.gate_norm = gate_norm
        n_extra = nh - 1
        self.W_k = nn.ModuleList([nn.Linear(n_neurons, d_mem, bias=False) for _ in range(n_extra)])
        self.W_v = nn.ModuleList([nn.Linear(n_neurons, d_mem, bias=False) for _ in range(n_extra)])
        self.W_b = nn.ModuleList([nn.Linear(n_neurons, d_mem) for _ in range(n_extra)])
        self.W_w = nn.ModuleList([nn.Linear(n_neurons, d_mem) for _ in range(n_extra)])
        self.W_fdecay = nn.ModuleList([nn.Linear(n_neurons, d_mem) for _ in range(n_extra)])
        self.gdn2_a = nn.ParameterList([nn.Parameter(torch.zeros(d_mem)) for _ in range(n_extra)])
        for lin in self.W_fdecay:
            with torch.no_grad():
                lin.weight.mul_(0.01)
                lin.bias.fill_(-3.0)
        if gate_norm and norm_ctor is not None:
            self.gate_norm_b = nn.ModuleList([norm_ctor(d_mem) for _ in range(n_extra)])
            self.gate_norm_w = nn.ModuleList([norm_ctor(d_mem) for _ in range(n_extra)])
            self.gate_norm_f = nn.ModuleList([norm_ctor(d_mem) for _ in range(n_extra)])
            self.gate_bias_f = nn.ParameterList([nn.Parameter(torch.full((d_mem,), -3.0)) for _ in range(n_extra)])

    def _gates(self, j, spk_seq, prev_for_k, spk_for_qv):
        k = self.W_k[j](prev_for_k)
        v = self.W_v[j](spk_for_qv)
        b_logit = self.W_b[j](spk_seq)
        w_logit = self.W_w[j](spk_seq)
        f_logit = self.W_fdecay[j](spk_seq)
        if self.gate_norm:
            b_logit = self.gate_norm_b[j](b_logit)
            w_logit = self.gate_norm_w[j](w_logit)
            f_logit = self.gate_norm_f[j](f_logit) + self.gate_bias_f[j]
        b = torch.sigmoid(b_logit)
        w = self.beta_floor + (1 - self.beta_floor) * torch.sigmoid(w_logit)
        alpha = torch.exp(-torch.exp(self.gdn2_a[j]) * F.softplus(f_logit))
        return v, k, b, w, alpha


def deltaproduct_gdn2_readout(dp, kernel_fn, spk_seq, prev_for_k, spk_for_qv,
                              v1, k1, b1, w1, alpha1, q_seq, M_init,
                              chunk, ckpt_chunks):
    """DeltaProduct via sequence expansion + the GDN-2 chunked kernel.

    v1..alpha1 are sub-step 1 ALREADY-TRANSFORMED tensors (from the block main
    path, reused verbatim). dp supplies sub-steps 2..n_h. Returns
    (r_out (B,T,D), M_final), r_out read at each token last sub-step.
    """
    B, T, D = v1.shape
    nh = dp.nh
    vs, ks, bs, ws, als = [v1], [k1], [b1], [w1], [alpha1]
    for j in range(nh - 1):
        v, k, b, w, a = dp._gates(j, spk_seq, prev_for_k, spk_for_qv)
        vs.append(v); ks.append(k); bs.append(b); ws.append(w); als.append(a)

    def interleave(seqs):
        return torch.stack(seqs, dim=2).reshape(B, T * nh, D)

    v_e = interleave(vs); k_e = interleave(ks)
    b_e = interleave(bs); w_e = interleave(ws); a_e = interleave(als)
    q_e = torch.zeros(B, T * nh, D, dtype=q_seq.dtype, device=q_seq.device)
    q_e[:, nh - 1::nh, :] = q_seq
    r_e, M = kernel_fn(v_e, k_e, q_e, b_e, w_e, a_e, M_init,
                       chunk=chunk, ckpt_chunks=ckpt_chunks)
    r_out = r_e.reshape(B, T, nh, D)[:, :, nh - 1, :]
    return r_out, M
