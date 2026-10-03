"""mk_r17_seeds.py -- 2026-10-03 (Glenn OK, overnight): DSN decision needs 2 seeds + one combination.
s1 repeats of r17f/g/i/j (identical to s0 except seed) and r17k = r17f (k 688) + r17g (ramp from 30%), s0 and s1.
Kept on the r17 recipe (no r18 speed flags) so they compare 1:1 with r17 s0."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
def emit(src, name, note, **upd):
    c = yaml.safe_load(open(R + src + ".yaml")); c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r17_seeds.py (2026-10-03) from {src}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)
for s in ["r17f_k688", "r17g_ramp03", "r17i_k512_ramp04", "r17j_x2_target20"]:
    emit(f"screen_{s}_s0_30m", f"screen_{s}_s1_30m", "Seed 1 repeat.", seed=1)
emit("screen_r17f_k688_s0_30m", "screen_r17k_k688_ramp03_s0_30m", "k 688 + ramp from 30%.", attn_ffn_spikeglu_ramp_start=0.3)
emit("screen_r17f_k688_s0_30m", "screen_r17k_k688_ramp03_s1_30m", "k 688 + ramp from 30%, seed 1.",
     attn_ffn_spikeglu_ramp_start=0.3, seed=1)
