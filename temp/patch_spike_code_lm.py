"""patch_spike_code_lm.py -- 2026-10-06: LM-level spike_code wiring (attention blocks, LM head input, memory-block input)."""
P = "/home/glenn/projects/neuro/src/model.py"
s = open(P).read()
rep = [
    ("                 mem_read_layers=None):",
     "                 mem_read_layers=None, spike_code=None, spike_levels=4, spike_levels_hidden=8):"),
    ("                    ffn_act=attn_ffn_act, ffn_homeo=attn_ffn_homeo))",
     "                    ffn_act=attn_ffn_act, ffn_homeo=attn_ffn_homeo,\n"
     "                    spike_code=[x for x in (spike_code or []) if x in (\"qkv\", \"o\", \"up\", \"down\")] or None,\n"
     "                    spike_levels=spike_levels, spike_levels_hidden=spike_levels_hidden))"),
    ("        self.mem_read_proj = None\n",
     "        # spike_code (2026-10-06 energy plan; None = off; created last): \"head\" codes the LM-head input, \"mem\" the\n"
     "        # input current of every spiking memory block (to_current); attention-block sites are passed above.\n"
     "        self.sc_head = None\n"
     "        if spike_code:\n"
     "            from spike_code import SpikeCode\n"
     "            if \"head\" in spike_code:\n"
     "                self.sc_head = SpikeCode(d_mem, spike_levels, signed=True, init=0.8)\n"
     "            if \"mem\" in spike_code:\n"
     "                for b in self.blocks:\n"
     "                    if isinstance(b, SpikingHebbianBlock):\n"
     "                        b.sc_in = SpikeCode(b.to_current.in_features, spike_levels, signed=True, init=0.8)\n"
     "        self.mem_read_proj = None\n"),
    ("            logits = self.head(x_seq)                          # (B, T, vocab)",
     "            logits = self.head(self.sc_head(x_seq) if self.sc_head is not None else x_seq)   # (B, T, vocab)"),
    ("        cur_seq = self.to_current(x_cur)                       # (B, T, N)",
     "        _sc = getattr(self, \"sc_in\", None)                     # spike_code \"mem\" (default off)\n"
     "        cur_seq = self.to_current(_sc(x_cur) if _sc is not None else x_cur)   # (B, T, N)"),
]
for a, b in rep:
    assert s.count(a) == 1, (s.count(a), a)
    s = s.replace(a, b)
open(P, "w").write(s)
print("[OK] model patched")
