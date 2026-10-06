"""patch_gap_flags.py -- 2026-09-28: add the ppl-gap plan flags (docs/ppl_gap_plan_2026_09_27.md) to
src/model.py + src/train_gpu.py. All default off (parity: temp/parity_gap_flags.py).
  python temp/patch_gap_flags.py <out_dir>   (reads src/, writes patched model.py/train_gpu.py into out_dir)"""
import sys, os
SRC = "/home/glenn/projects/neuro/src/"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def rep(s, old, new, count=1):
    n = s.count(old)
    assert n == count, (old[:80], n)
    return s.replace(old, new)

m = open(SRC + "model.py", encoding="utf-8").read()
# R1 block signature
m = rep(m, "                 gate_rank=0, lif_exact=False, lif_reset=1.0):\n        super().__init__()\n        self.n_neurons = n_neurons",
        "                 gate_rank=0, lif_exact=False, lif_reset=1.0,\n"
        "                 ff_conv=0, swa_window=0, swa_heads=4, swa_dim=64, pre_norm=False,\n"
        "                 mem_out_gate=0, lif_levels=1, lif_signed=False, lif_learn_beta=False):\n"
        "        super().__init__()\n        self.n_neurons = n_neurons")
# R2 new modules at the end of the block __init__
m = rep(m, """            with torch.no_grad():
                out_lin.weight.mul_(depth_scale)

    def _lif_step(self, cur, mem):""", """            with torch.no_grad():
                out_lin.weight.mul_(depth_scale)

        # --- ppl-gap plan flags (2026-09-28, docs/ppl_gap_plan_2026_09_27.md). All default off: no
        # params are created and forward_sequence is unchanged (temp/parity_gap_flags.py). ---
        # B1 ff_conv=k: causal depthwise conv over the spike sequence feeding ONLY W_ff, as a residual
        # spk + conv(spk) with zero-init taps (exact control at init; weight decay pulls toward it).
        # The memory's k/q/v and the gates keep the clean spikes, so the recall pointer is untouched
        # (the 09-07 pre_conv blurred k/q and killed recall).
        self.ff_conv = ff_conv
        if ff_conv:
            self.ff_conv_d = nn.Parameter(torch.zeros(n_neurons, 1, ff_conv))
        # B2 swa_window=W: position-aware local attention read. q/k from the CURRENT spikes via small
        # projections (swa_heads x swa_dim), RoPE, parameter-free RMS QK-norm, causal window W; values =
        # the memory's v_seq split into heads. Added to the memory read through a per-channel gate
        # initialised at 0 (starts as the control). Unlike local_attn (09-07 NULL) it has positions.
        self.swa_window = swa_window
        if swa_window:
            if d_mem % swa_heads:
                raise ValueError(f"d_mem ({d_mem}) must be divisible by swa_heads ({swa_heads})")
            from hybrid_attn import _rope_freqs
            self.swa_heads, self.swa_dim = swa_heads, swa_dim
            self.W_qs = nn.Linear(n_neurons, swa_heads * swa_dim, bias=False)
            self.W_ks = nn.Linear(n_neurons, swa_heads * swa_dim, bias=False)
            self.swa_gate = nn.Parameter(torch.zeros(d_mem))
            cos, sin = _rope_freqs(swa_dim, 4096)
            self.register_buffer("swa_cos", cos, persistent=False)
            self.register_buffer("swa_sin", sin, persistent=False)
        # D1 pre_norm: RMSNorm on the block input before to_current (standard pre-norm).
        self.pre_norm = RMSNorm(d_in) if pre_norm else None
        # D2 mem_out_gate=r: data-dependent output gate on the memory read, r <- r * 2*sigmoid(W_og(spk)),
        # rank-r bottleneck, up-projection zero-init -> gate == 1 at init (GDN / Gated-Attention style).
        self.mem_out_gate = mem_out_gate
        if mem_out_gate:
            self.W_og = LowRankLinear(n_neurons, d_mem, mem_out_gate)
            with torch.no_grad():
                self.W_og.up.weight.zero_()
                self.W_og.up.bias.zero_()
        # C1/C2/C3 neuron (src/lif_exact.py lif_exact_ml): integer multi-level spikes (lif_levels),
        # signed/ternary spikes (lif_signed), learnable per-neuron membrane decay (lif_learn_beta,
        # init = beta). lif_exact kernel only.
        self.lif_levels, self.lif_signed, self.lif_learn_beta = lif_levels, lif_signed, lif_learn_beta
        self.lif_ml = lif_levels > 1 or lif_signed or lif_learn_beta
        if self.lif_ml and not lif_exact:
            raise ValueError("lif_levels/lif_signed/lif_learn_beta require lif_exact: true")
        if lif_learn_beta:
            self.beta_logit = nn.Parameter(torch.full((n_neurons,), math.log(beta / (1.0 - beta))))

    def _lif_step(self, cur, mem):""")
# R7 SWA read helper, placed right before forward_sequence
m = rep(m, "    def forward_sequence(self, x_seq, mem, M, prev_spk, ablate_memory=False,\n                         fpt_K=10):",
        '''    def _swa_read(self, spk_seq, v_seq):
        """B2: causal sliding-window softmax attention read (see __init__). Returns gate * (B, T, d_mem)."""
        from hybrid_attn import _apply_rope
        B, T, _ = spk_seq.shape
        H, Dh = self.swa_heads, self.swa_dim
        q = self.W_qs(spk_seq).view(B, T, H, Dh).transpose(1, 2)
        k = self.W_ks(spk_seq).view(B, T, H, Dh).transpose(1, 2)
        v = v_seq.reshape(B, T, H, -1).transpose(1, 2)
        q = _apply_rope(F.rms_norm(q, (Dh,)), self.swa_cos, self.swa_sin)
        k = _apply_rope(F.rms_norm(k, (Dh,)), self.swa_cos, self.swa_sin)
        idx = torch.arange(T, device=spk_seq.device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        band = (diff >= 0) & (diff < self.swa_window)                  # (T, T) query x key
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=band)
        return o.transpose(1, 2).reshape(B, T, -1) * self.swa_gate.to(o.dtype)

    def forward_sequence(self, x_seq, mem, M, prev_spk, ablate_memory=False,
                         fpt_K=10):''')
# R3 LIF selection + pre-norm
m = rep(m, """        if self.lif_exact:
            import functools
            from lif_exact import lif_exact
            lif_parallel = functools.partial(lif_exact, gamma=self.lif_reset)
        else:
            from lif_parallel import lif_parallel
        B, T, _ = x_seq.shape
        cur_seq = self.to_current(x_seq)                       # (B, T, N)""",
        """        if self.lif_exact:
            import functools
            from lif_exact import lif_exact, lif_exact_ml
            if self.lif_ml:
                lif_parallel = functools.partial(lif_exact_ml, gamma=self.lif_reset,
                                                 levels=self.lif_levels, signed=self.lif_signed)
            else:
                lif_parallel = functools.partial(lif_exact, gamma=self.lif_reset)
        else:
            from lif_parallel import lif_parallel
        beta_arg = torch.sigmoid(self.beta_logit) if self.lif_learn_beta else self.beta_val
        x_cur = x_seq if self.pre_norm is None else self.pre_norm(x_seq)
        B, T, _ = x_seq.shape
        cur_seq = self.to_current(x_cur)                       # (B, T, N)""")
# R4 beta argument (both LIF calls) + recurrent current
m = rep(m, "        spk_seq, mem_final = lif_parallel(cur_seq, self.beta_val, vth=1.0,",
        "        spk_seq, mem_final = lif_parallel(cur_seq, beta_arg, vth=1.0,")
m = rep(m, "            spk_seq, mem_final = lif_parallel(cur_seq2, self.beta_val, vth=1.0,",
        "            spk_seq, mem_final = lif_parallel(cur_seq2, beta_arg, vth=1.0,")
m = rep(m, "            cur_seq2 = self.to_current(x_seq) + F.linear(",
        "            cur_seq2 = self.to_current(x_cur) + F.linear(")
# R5 ff_conv
m = rep(m, "        ff_seq = self.W_ff(spk_seq)\n",
        """        if self.ff_conv:
            spk_ff = spk_seq + F.conv1d(F.pad(spk_seq.transpose(1, 2), (self.ff_conv - 1, 0)),
                                        self.ff_conv_d.to(spk_seq.dtype),
                                        groups=self.n_neurons).transpose(1, 2)
            ff_seq = self.W_ff(spk_ff)
        else:
            ff_seq = self.W_ff(spk_seq)
""")
# R6 output gate + SWA read
m = rep(m, """            out_seq = self.norm(r_seq + ff_seq)
        else:
            out_seq = self.norm(ff_seq)
        final_spk = spk_seq[:, -1, :]""", """            if self.mem_out_gate:
                r_seq = r_seq * (2.0 * torch.sigmoid(self.W_og(spk_seq)))
            if self.swa_window:
                r_seq = r_seq + self._swa_read(spk_seq, v_seq)
            out_seq = self.norm(r_seq + ff_seq)
        elif self.swa_window:
            out_seq = self.norm(ff_seq + self._swa_read(spk_seq, v_seq))
        else:
            out_seq = self.norm(ff_seq)
        final_spk = spk_seq[:, -1, :]""")
# R8 LM signature
m = rep(m, "                 deltaproduct_nh=1, gate_rank=0, lif_exact=False, lif_reset=1.0, act_ckpt=False,",
        "                 deltaproduct_nh=1, gate_rank=0, lif_exact=False, lif_reset=1.0, act_ckpt=False,\n"
        "                 ff_conv=0, swa_window=0, swa_heads=4, swa_dim=64, pre_norm=False,\n"
        "                 mem_out_gate=0, lif_levels=1, lif_signed=False, lif_learn_beta=False,")
# R9 block construction
m = rep(m, "                    gate_rank=gate_rank, lif_exact=lif_exact, lif_reset=lif_reset))",
        "                    gate_rank=gate_rank, lif_exact=lif_exact, lif_reset=lif_reset,\n"
        "                    ff_conv=ff_conv, swa_window=swa_window, swa_heads=swa_heads, swa_dim=swa_dim,\n"
        "                    pre_norm=pre_norm, mem_out_gate=mem_out_gate, lif_levels=lif_levels,\n"
        "                    lif_signed=lif_signed, lif_learn_beta=lif_learn_beta))")
# R10 spike rate = mean |s| for signed spikes
m = rep(m, "        self._spike_rate_undiluted = spike_rate_undiluted\n",
        "        self._spike_rate_undiluted = spike_rate_undiluted\n"
        "        self._spk_abs = lif_signed   # signed spikes: rate = mean |s| (else +-1 would cancel)\n")
m = rep(m, "[s.mean() for s in all_spikes]", "[(s.abs() if self._spk_abs else s).mean() for s in all_spikes]", count=2)
open(os.path.join(out, "model.py"), "w", encoding="utf-8", newline="").write(m)

t = open(SRC + "train_gpu.py", encoding="utf-8").read()
t = rep(t, '                                lif_reset=float(c.get("lif_reset", 1.0)),\n',
        '                                lif_reset=float(c.get("lif_reset", 1.0)),\n'
        '                                ff_conv=c.get("ff_conv", 0),\n'
        '                                swa_window=c.get("swa_window", 0),\n'
        '                                swa_heads=c.get("swa_heads", 4),\n'
        '                                swa_dim=c.get("swa_dim", 64),\n'
        '                                pre_norm=c.get("pre_norm", False),\n'
        '                                mem_out_gate=c.get("mem_out_gate", 0),\n'
        '                                lif_levels=c.get("lif_levels", 1),\n'
        '                                lif_signed=c.get("lif_signed", False),\n'
        '                                lif_learn_beta=c.get("lif_learn_beta", False),\n')
open(os.path.join(out, "train_gpu.py"), "w", encoding="utf-8", newline="").write(t)
print("[OK] patched model.py + train_gpu.py ->", out)
