"""patch_gap_flags3.py -- 2026-09-28 round-6 flags (docs/ppl_gap_plan_2026_09_27.md), default off:
  mqar_gap_max=G (train_gpu): HAYSTACK MQAR -- [k1 v1 .. kN vN] + g tokens of real TRAIN text + [kq], g ~ U[0, cur_max],
      cur_max ramps linearly 1 -> G over the first mqar_gap_ramp_frac of training (0 = no ramp). "Distance curriculum" that
      breaks the fixed-state recall "interference wall" (arXiv 2609.16183). Gap text uses a PRIVATE numpy RNG, so the LM
      data stream is byte-identical to the control; mqar_batch (default batch_size) sets the MQAR micro-batch.
      Why: haystack_recall_eval (logs/haystack_recall_r4.log) -- best-ppl full stack recall N8 falls 100% -> 8% with 64
      tokens of real text between the facts and the query; our MQAR aux never has a gap.
  canon_a=k (model): Canon-A (Allen-Zhu, Physics of LM 4.1): residual depthwise causal conv on each spiking block's
      input, x + conv_k(x), zero-init (= control at init), feeding to_current only.
  python temp/patch_gap_flags3.py <out_dir>"""
import sys, os
SRC = "/home/glenn/projects/neuro/src/"; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a)); return s.replace(a, b)
m = open(SRC + "model.py", encoding="utf-8").read()
m = rep(m, "                 lif_alpha=None, spk_mlp=0):\n        super().__init__()",
        "                 lif_alpha=None, spk_mlp=0, canon_a=0):\n        super().__init__()")
m = rep(m, """            with torch.no_grad():
                self.W_down.weight.zero_()
""", """            with torch.no_grad():
                self.W_down.weight.zero_()
        # Round 6 canon_a=k: Canon-A residual causal depthwise conv on the block input (zero-init taps).
        self.canon_a = canon_a
        if canon_a:
            self.canon_a_w = nn.Parameter(torch.zeros(d_in, 1, canon_a))
""")
m = rep(m, "        x_cur = x_seq if self.pre_norm is None else self.pre_norm(x_seq)\n",
        """        x_cur = x_seq
        if self.canon_a:
            x_cur = x_seq + F.conv1d(F.pad(x_seq.transpose(1, 2), (self.canon_a - 1, 0)),
                                     self.canon_a_w.to(x_seq.dtype), groups=x_seq.size(-1)).transpose(1, 2)
        x_cur = x_cur if self.pre_norm is None else self.pre_norm(x_cur)
""")
m = rep(m, "                 lif_alpha=None, spk_mlp=0,\n                 attention_layer_positions=None,",
        "                 lif_alpha=None, spk_mlp=0, canon_a=0,\n                 attention_layer_positions=None,")
m = rep(m, "                    lif_alpha=lif_alpha, spk_mlp=spk_mlp))",
        "                    lif_alpha=lif_alpha, spk_mlp=spk_mlp, canon_a=canon_a))")
open(os.path.join(out, "model.py"), "w", encoding="utf-8", newline="").write(m)

t = open(SRC + "train_gpu.py", encoding="utf-8").read()
t = rep(t, '                                spk_mlp=c.get("spk_mlp", 0),\n',
        '                                spk_mlp=c.get("spk_mlp", 0),\n'
        '                                canon_a=c.get("canon_a", 0),\n')
t = rep(t, "    return torch.stack(xs).to(device), torch.tensor(tgs, device=device)\n",
        '''    return torch.stack(xs).to(device), torch.tensor(tgs, device=device)


def mqar_gap_batch(B, N, g, data, rng, device):
    """Haystack MQAR (2026-09-28; 'interference wall', arXiv 2609.16183): [k1 v1 .. kN vN] + g tokens of real TRAIN
    text + [kq] -> target vq. Gap windows come from the private numpy RNG `rng` (LM data stream untouched); gap tokens
    equal to one of the sequence's keys are replaced by random ids in 6000-7999 (outside the key/value ranges)."""
    x, tg = mqar_aug_batch(B, N, "cpu")                      # (B, 2N+1) = kv block + query
    if g > 0:
        starts = rng.randint(0, data.split - g - 1, size=B)
        gap = torch.from_numpy(data.lut[np.stack([np.asarray(data.data[s:s + g], dtype=np.int64) for s in starts])])
        keys = x[:, 0:2 * N:2]
        clash = (gap.unsqueeze(-1) == keys.unsqueeze(1)).any(-1)
        gap = torch.where(clash, torch.randint(6000, 8000, gap.shape), gap)
        x = torch.cat([x[:, :2 * N], gap, x[:, 2 * N:]], 1)
    return x.to(device), tg.to(device)
''')
t = rep(t, "    mqar_model = raw_model\n",
        "    mqar_model = raw_model\n"
        "    _mqar_gap_rng = np.random.RandomState(int(c.get(\"seed\", 0)) + 7919)   # haystack MQAR only (mqar_gap_max)\n")
t = rep(t, "                    x_mq, tgt_mq = mqar_aug_batch(c[\"batch_size\"], N_mq, device)\n",
        """                    _gmax = int(c.get("mqar_gap_max", 0))
                    if _gmax > 0:
                        _rf = float(c.get("mqar_gap_ramp_frac", 0.0))
                        _cur = _gmax if _rf <= 0 else max(1, int(_gmax * min(1.0, step / max(1.0, _rf * total_steps))))
                        g_mq = int(_mqar_gap_rng.randint(0, _cur + 1))
                        x_mq, tgt_mq = mqar_gap_batch(int(c.get("mqar_batch", c["batch_size"])), N_mq, g_mq,
                                                      data, _mqar_gap_rng, device)
                    else:
                        x_mq, tgt_mq = mqar_aug_batch(c["batch_size"], N_mq, device)
""")
open(os.path.join(out, "train_gpu.py"), "w", encoding="utf-8", newline="").write(t)
print("[OK] ->", out)
