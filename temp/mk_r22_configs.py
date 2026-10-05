"""mk_r22_configs.py -- 2026-10-04: 30M side screens on the 5080 (the G3-shape arms don't fit in 16 GB).
Base = r19e (sparse MLP 5%, 4L, 30M rig). Does the x1 late gap dose keep haystack ~100% with the sparse MLP, and what
does a fixed 128-token attention window do at 30M (memory must carry beyond 128)? Eager (as the 30M rig)."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
SRC = "screen_r19e_layer05_s0_30m"
X1 = dict(mqar_gap_max=440, mqar_gap_start_frac=0.7, mqar_gap_ramp_frac=0.1, mqar_gap_weight=0.15, mqar_batch=16)
for name, upd, note in [
    ("screen_r22a_x1_s0_30m", X1, "r19e + x1 late gap dose."),
    ("screen_r22b_w128x1_s0_30m", dict(X1, attn_window_fixed=128), "r19e + fixed 128 window + x1."),
]:
    c = yaml.safe_load(open(R + SRC + ".yaml")); c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r22_configs.py (2026-10-04) from {SRC}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)
