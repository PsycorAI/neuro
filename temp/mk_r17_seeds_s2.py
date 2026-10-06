"""mk_r17_seeds_s2.py -- 2026-10-03 02:05 (Glenn: "utilize the 5090 and 5080 tonight"): fills the GPU tail after the
night2 queues (~04:30). DSN decision gets a 3rd seed for c0 / r17f / r17i / r17k, and the dense reference l3 GELU
(40.13, 1 seed only) gets seed 1. Identical to the source configs except seed (r17 recipe, 1:1 compare)."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
def emit(src, name, note, **upd):
    c = yaml.safe_load(open(R + src + ".yaml")); c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r17_seeds_s2.py (2026-10-03) from {src}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)
for s in ["r16_c0", "r17f_k688", "r17i_k512_ramp04", "r17k_k688_ramp03"]:
    emit(f"screen_{s}_s0_30m", f"screen_{s}_s2_30m", "Seed 2 repeat.", seed=2)
emit("screen_r16_l3_gelu_s0_30m", "screen_r16_l3_gelu_s1_30m", "Seed 1 repeat.", seed=1)
