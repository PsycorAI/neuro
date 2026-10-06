"""patch_gap_flags2.py -- 2026-09-28 round-5 ppl-gap flags (docs/ppl_gap_plan_2026_09_27.md), default off:
  lif_alpha  : surrogate-gradient width for the lif_exact kernels (None = kernel default 2.0 = unchanged)
  spk_mlp=M  : D3 spiking MLP sub-block inside each spiking block: out += W_down(LIF(W_up(RMSNorm(x + out)))),
               M neurons, same neuron type as the block (incl. lif_levels), W_down zero-init (= control at init).
               Its spikes are appended to the block's spike stats so the sparsity penalty covers them.
  python temp/patch_gap_flags2.py <out_dir>   (reads src/, writes patched model.py/train_gpu.py into out_dir)"""
import sys, os
SRC = "/home/glenn/projects/neuro/src/"; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a)); return s.replace(a, b)
m = open(SRC + "model.py", encoding="utf-8").read()
m = rep(m, "                 mem_out_gate=0, lif_levels=1, lif_signed=False, lif_learn_beta=False):\n        super().__init__()",
        "                 mem_out_gate=0, lif_levels=1, lif_signed=False, lif_learn_beta=False,\n"
        "                 lif_alpha=None, spk_mlp=0):\n        super().__init__()")
m = rep(m, """        if lif_learn_beta:
            self.beta_logit = nn.Parameter(torch.full((n_neurons,), math.log(beta / (1.0 - beta))))
""", """        if lif_learn_beta:
            self.beta_logit = nn.Parameter(torch.full((n_neurons,), math.log(beta / (1.0 - beta))))
        # Round 5 (2026-09-28). lif_alpha: surrogate width (lif_exact only; None = kernel default 2.0).
        self.lif_alpha = lif_alpha
        if lif_alpha is not None and not lif_exact:
            raise ValueError("lif_alpha requires lif_exact: true")
        # D3 spk_mlp=M: spiking MLP sub-block (pre-norm, M neurons, W_down zero-init -> control at init).
        self.spk_mlp = spk_mlp
        if spk_mlp:
            if d_in != d_mem:
                raise ValueError("spk_mlp requires d_in == d_mem (residual inside the block)")
            self.mlp_norm = RMSNorm(d_mem)
            self.W_up = nn.Linear(d_mem, spk_mlp)
            self.W_down = nn.Linear(spk_mlp, d_mem, bias=False)
            with torch.no_grad():
                self.W_down.weight.zero_()
""")
m = rep(m, """            if self.lif_ml:
                lif_parallel = functools.partial(lif_exact_ml, gamma=self.lif_reset,
                                                 levels=self.lif_levels, signed=self.lif_signed)
            else:
                lif_parallel = functools.partial(lif_exact, gamma=self.lif_reset)""",
        """            akw = {} if self.lif_alpha is None else {"alpha": self.lif_alpha}
            if self.lif_ml:
                lif_parallel = functools.partial(lif_exact_ml, gamma=self.lif_reset,
                                                 levels=self.lif_levels, signed=self.lif_signed, **akw)
            else:
                lif_parallel = functools.partial(lif_exact, gamma=self.lif_reset, **akw)""")
m = rep(m, """        else:
            out_seq = self.norm(ff_seq)
        final_spk = spk_seq[:, -1, :]
        return out_seq, mem_final, M, spk_seq, final_spk""", """        else:
            out_seq = self.norm(ff_seq)
        final_spk = spk_seq[:, -1, :]
        if self.spk_mlp:
            spk_m, _ = lif_parallel(self.W_up(self.mlp_norm(x_seq + out_seq)), self.beta_val,
                                    vth=1.0, K=fpt_K)
            out_seq = out_seq + self.W_down(spk_m)
            spk_seq = torch.cat([spk_seq, spk_m], dim=-1)   # stats only (sparsity penalty covers both)
        return out_seq, mem_final, M, spk_seq, final_spk""")
m = rep(m, "                 mem_out_gate=0, lif_levels=1, lif_signed=False, lif_learn_beta=False,\n                 attention_layer_positions=None,",
        "                 mem_out_gate=0, lif_levels=1, lif_signed=False, lif_learn_beta=False,\n"
        "                 lif_alpha=None, spk_mlp=0,\n                 attention_layer_positions=None,")
m = rep(m, "                    lif_signed=lif_signed, lif_learn_beta=lif_learn_beta))",
        "                    lif_signed=lif_signed, lif_learn_beta=lif_learn_beta,\n"
        "                    lif_alpha=lif_alpha, spk_mlp=spk_mlp))")
open(os.path.join(out, "model.py"), "w", encoding="utf-8", newline="").write(m)
t = open(SRC + "train_gpu.py", encoding="utf-8").read()
t = rep(t, '                                lif_learn_beta=c.get("lif_learn_beta", False),\n',
        '                                lif_learn_beta=c.get("lif_learn_beta", False),\n'
        '                                lif_alpha=c.get("lif_alpha", None),\n'
        '                                spk_mlp=c.get("spk_mlp", 0),\n')
open(os.path.join(out, "train_gpu.py"), "w", encoding="utf-8", newline="").write(t)
print("[OK] ->", out)
