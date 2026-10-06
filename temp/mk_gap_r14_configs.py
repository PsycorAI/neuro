"""mk_gap_r14_configs.py -- 2026-10-01 round 14 (Glenn OK): build on r13a = 3L [spk, attn, attn] (51.24, 0.86x tfmr 3L).
  r14a: r13a seed 1                                   -- confirm the win
  r14b: r13a + canon_a 4 + late ramp to 10%           -- stack the round-8/11 levers (r11c: 60.19 @ 9.5% on 2L)
  r14c: r14b + gap 440 at mqar_weight 0.15            -- candidate v4 recipe (haystack)
  r14d: 4L [spk, attn, attn, attn]                    -- does more attention keep paying?"""
import re
D = "/home/glenn/projects/neuro/configs/"
SRC = "screen_r13a_signed_3L_attn2_s0_30m"
def make(name, why, edits, append=""):
    s = open(D + SRC + ".yaml").read()
    s = re.sub(r"^(# .*\n)+", "", s, count=1)
    s = f"# {name} -- round 14 (2026-10-01). AUTO-GEN temp/mk_gap_r14_configs.py from {SRC} (51.24). Change: {why}\n" + s
    for pat, rep in edits + [(rf"^run_name: {SRC}$", f"run_name: {name}")]:
        s, n = re.subn(pat, rep, s, flags=re.M)
        assert n == 1, (name, pat)
    s = s.rstrip("\n") + "\n" + append
    open(D + name + ".yaml", "w").write(s)
    print("[OK]", name)
RAMP10 = ("canon_a: 4\n"
          "sparsity_ramp_start_frac: 0.6    # lambda 0.5 until 60% of training, then linear to 8.0 (as r11c)\n"
          "sparsity_lambda_final: 8.0\n")
GAP = "mqar_gap_max: 440\nmqar_gap_ramp_frac: 0.3\nmqar_batch: 16\n"
make("screen_r14a_3Lattn2_s1_30m", "seed 1.", [(r"^seed: 0$", "seed: 1")])
make("screen_r14b_3Lattn2_canon_ramp10_s0_30m", "+ canon_a 4 + late sparsity ramp to 10% (target 0.10, lambda 0.5 -> 8.0 from 60%).",
     [(r"^sparsity_target: 0\.05$", "sparsity_target: 0.10")], RAMP10)
make("screen_r14c_3Lattn2_canon_ramp10_gapw015_s0_30m", "r14b + haystack MQAR gap 440 at mqar_weight 0.15.",
     [(r"^sparsity_target: 0\.05$", "sparsity_target: 0.10"), (r"^mqar_weight: 0\.5$", "mqar_weight: 0.15")], RAMP10 + GAP)
make("screen_r14d_4L_attn3_s0_30m", "4L [spk, attn, attn, attn] (attention_layer_positions [1, 2, 3]).",
     [(r"^n_layers: 3$", "n_layers: 4"), (r"^attention_layer_positions:\n- 1\n- 2$", "attention_layer_positions:\n- 1\n- 2\n- 3")])
