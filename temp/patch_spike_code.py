"""patch_spike_code.py -- 2026-10-06: route qkv / o_proj / MLP inputs through self.sc codes (train + decode paths)."""
P = "/home/glenn/projects/neuro/src/hybrid_attn.py"
s = open(P).read()
rep = [
    ("        qkv = self.qkv(x)                                        # (B, T, 3D)",
     "        qkv = self.qkv(self.sc[\"qkv\"](x) if \"qkv\" in self.sc else x)   # (B, T, 3D)"),
    ("        return self.o_proj(out)\n",
     "        return self.o_proj(self.sc[\"o\"](out) if \"o\" in self.sc else out)\n"),
    ("        qkv = self.qkv(x)\n",
     "        qkv = self.qkv(self.sc[\"qkv\"](x) if \"qkv\" in self.sc else x)\n"),
    ("        return self.o_proj(out), new_cache",
     "        return self.o_proj(self.sc[\"o\"](out) if \"o\" in self.sc else out), new_cache"),
    ("        return y if self.ffn is None else y + self.ffn(self.norm_ffn(y))",
     "        return y if self.ffn is None else y + self._ffn(self.norm_ffn(y))"),
    ("        z = y if self.ffn is None else y + self.ffn(self.norm_ffn(y))",
     "        z = y if self.ffn is None else y + self._ffn(self.norm_ffn(y))"),
    ("    def _block(self, x_seq):",
     "    def _ffn(self, h):\n"
     "        if not (\"up\" in self.sc or \"down\" in self.sc):\n"
     "            return self.ffn(h)\n"
     "        if \"up\" in self.sc:\n"
     "            h = self.sc[\"up\"](h)\n"
     "        h = self.ffn[1](self.ffn[0](h))\n"
     "        if \"down\" in self.sc:\n"
     "            h = self.sc[\"down\"](h)\n"
     "        return self.ffn[2](h)\n\n"
     "    def _block(self, x_seq):"),
]
for a, b in rep:
    assert s.count(a) == 1, (s.count(a), a)
    s = s.replace(a, b)
open(P, "w").write(s)
print("[OK] hybrid_attn patched")
