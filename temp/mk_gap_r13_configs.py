"""mk_gap_r13_configs.py -- 2026-09-30 round 13: does a 3rd layer pay off for the signed stack? (to-do DEPTH WARNING)
2L->3L: tfmr 68.8->59.38, unsigned 74.9->66.72, signed 61.24->63.35/64.65 (worse). All seed 0 (vs 3L s0 64.65, 2L s0 61.36).
  r13a: 3L [spk, attn, attn]      -- depth from attention, ONE GDN memory (74.2M params)
  r13b: 3L [spk, attn, spk] + depth_scaled_init -- is the slow start an init problem? (72.3M)
  r13d: 2L signed n_neurons 4096  -- param-matched width control for the 3rd layer (72.2M; probe temp/probe_r13_params.py)"""
import re
D = "/home/glenn/projects/neuro/configs/"
def make(src, name, why, edits, append=""):
    s = open(D + src + ".yaml").read()
    s = re.sub(r"^(# .*\n)+", "", s, count=1)
    s = f"# {name} -- round 13 (2026-09-30). AUTO-GEN temp/mk_gap_r13_configs.py from {src}. Change: {why}\n" + s
    for pat, rep in edits + [(rf"^run_name: {src}$", f"run_name: {name}")]:
        s, n = re.subn(pat, rep, s, flags=re.M)
        assert n == 1, (name, pat)
    s = s.rstrip("\n") + "\n" + append
    open(D + name + ".yaml", "w").write(s)
    print("[OK]", name)
make("screen_r8_signed_3L_s0_30m", "screen_r13a_signed_3L_attn2_s0_30m", "3L [spk, attn, attn] (attention_layer_positions [1, 2]).",
     [(r"^attention_layer_positions:\n- 1$", "attention_layer_positions:\n- 1\n- 2")])
make("screen_r8_signed_3L_s0_30m", "screen_r13b_signed_3L_dsinit_s0_30m", "+ depth_scaled_init (residual out-proj x 1/sqrt(2L)).",
     [], "depth_scaled_init: true\n")
make("screen_r5_signed_s0_30m", "screen_r13d_signed_n4096_s0_30m", "n_neurons 2048 -> 4096 (param-matched to 3L signed, 72.2M).",
     [(r"^n_neurons: 2048$", "n_neurons: 4096")])
