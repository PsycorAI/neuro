"""patch_head_rank.py -- 2026-10-06: factorised LM head (head_rank r: d -> r -> V, default None = byte-identical)."""
P = "/home/glenn/projects/neuro/src/model.py"
s = open(P).read()
rep = [
    ("                 mem_read_layers=None, spike_code=None, spike_levels=4, spike_levels_hidden=8):",
     "                 mem_read_layers=None, spike_code=None, spike_levels=4, spike_levels_hidden=8, head_rank=None):"),
    ("        self.head = nn.Linear(d_mem, vocab)\n",
     "        # head_rank (2026-10-06 energy plan, default None = byte-identical): factorised head d -> r -> V (head_in then\n"
     "        # head), ~(d + V) * r ops instead of d * V; conversion runs initialise it from the trained head's SVD.\n"
     "        self.head_in = nn.Linear(d_mem, int(head_rank), bias=False) if head_rank else None\n"
     "        self.head = nn.Linear(int(head_rank) if head_rank else d_mem, vocab)\n"),
    ("            if \"head\" in spike_code:\n",
     "            if \"head\" in spike_code and self.head_in is not None:\n"
     "                self.sc_head_mid = SpikeCode(self.head_in.out_features, spike_levels, signed=True, init=0.8)\n"
     "            if \"head\" in spike_code:\n"),
    ("            logits = self.head(self.sc_head(x_seq) if self.sc_head is not None else x_seq)   # (B, T, vocab)",
     "            _h = self.sc_head(x_seq) if self.sc_head is not None else x_seq\n"
     "            if self.head_in is not None:\n"
     "                _h = self.head_in(_h)\n"
     "                if getattr(self, \"sc_head_mid\", None) is not None:\n"
     "                    _h = self.sc_head_mid(_h)\n"
     "            logits = self.head(_h)                                           # (B, T, vocab)"),
]
for a, b in rep:
    assert s.count(a) == 1, (s.count(a), a)
    s = s.replace(a, b)
open(P, "w").write(s)
P2 = "/home/glenn/projects/neuro/src/train_gpu.py"
t = open(P2).read()
a = "                                spike_levels_hidden=int(c.get(\"spike_levels_hidden\", 8)))"
assert t.count(a) == 1
t = t.replace(a, "                                spike_levels_hidden=int(c.get(\"spike_levels_hidden\", 8)),\n"
                 "                                head_rank=c.get(\"head_rank\"))")
open(P2, "w").write(t)
print("[OK] head_rank patched")
