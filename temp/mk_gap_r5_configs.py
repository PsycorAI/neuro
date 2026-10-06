"""mk_gap_r5_configs.py -- 2026-09-28 ppl-gap round 5 (docs/ppl_gap_plan_2026_09_27.md). New control = full stack
screen_gap_hyb_full_s0_30m (2L [spk,attn] + lif_levels 4 + pre_norm + mem_out_gate 16 = 73.44; tfmr 2L 68.5, 3L 59.38)."""
import re
C = "/home/glenn/projects/neuro/configs/"
src = open(C + "screen_gap_hyb_full_s0_30m.yaml", encoding="utf-8").read()
body = "\n".join(l for l in src.splitlines() if not l.startswith("#")) + "\n"
def setk(b, k, v):
    b2, n = re.subn(rf"^{k}:.*$", f"{k}: {v}", b, flags=re.M)
    return b2 if n else b + f"{k}: {v}\n"
ARMS = {
  "full_s1":   ("seed-1 of the full stack (confirm 73.44)", {"seed": 1}),
  "lev8":      ("lif_levels 8 (vs 4)", {"lif_levels": 8}),
  "lev16":     ("lif_levels 16 (vs 4)", {"lif_levels": 16}),
  "signed":    ("lif_signed (signed multi-level spikes -4..4)", {"lif_signed": "true"}),
  "spkmlp":    ("spk_mlp 2048: spiking MLP sub-block per spiking layer (+4.2M params/layer; NOT param-matched)", {"spk_mlp": 2048}),
  "alpha1":    ("lif_alpha 1.0 (wider surrogate; default 2.0)", {"lif_alpha": "1.0"}),
  "alpha4":    ("lif_alpha 4.0 (narrower surrogate)", {"lif_alpha": "4.0"}),
  "engram32k": ("engram_v2_K 32768 (4x n-gram table; +params)", {"engram_v2_K": 32768}),
  "mqar025":   ("mqar_weight 0.25 (recall saturated at 100%; RECIPE change -> the tfmr would need the same)", {"mqar_weight": 0.25}),
  "3L":        ("3 layers [spk, attn, spk], 1 attention layer (vs 3L tfmr 59.38)", {"n_layers": 3}),
  "3L_2attn":  ("3 layers [attn, spk, attn], 2 attention layers (vs 3L tfmr 59.38)", {"n_layers": 3, "attention_layer_positions": "[0, 2]"}),
}
for tag, (what, kv) in ARMS.items():
    n = f"screen_r5_{tag}_s0_30m" if tag != "full_s1" else "screen_r5_full_s1_30m"
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", body, flags=re.M)
    if "attention_layer_positions" in kv:
        b = re.sub(r"^attention_layer_positions:\n- 1\n", f"attention_layer_positions: {kv.pop('attention_layer_positions')}\n", b, flags=re.M)
    for k, v in kv.items():
        b = setk(b, k, v)
    hdr = (f"# {n} -- ppl-gap round 5 (2026-09-28). AUTO-GEN temp/mk_gap_r5_configs.py from screen_gap_hyb_full_s0_30m\n"
           f"# (full stack, 73.44). ONE change: {what}.\n")
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b)
    print("[OK]", n)
