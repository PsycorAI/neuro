"""mk_r23_configs.py -- 2026-10-04 (Glenn OK: "start on this plan and next tests"). Depth proxy for the
"hippocampus unused at depth" finding, sized for the 5080 (16 GB).
At 30M (d1024, 4L = H0 + 3 attn) memory carries recall (Δ +4.6, haystack 100%); at 150M (d768, 12L = H0 + 11 attn)
memory Δ = 0.00 even with w128 + x1 (r21c), while its write gates are mostly OPEN (28% at floor) -> later layers
ignore the memory output. Question: is it DEPTH (number of attention layers) or WIDTH/scale?
Same G3 width (d768, n3072) and recipe as r21i (x1 gap MQAR from step 0, full window), only the depth changes:
  r23a  4 layers (H0 + 3 attn)    r23b  6 layers (H0 + 5 attn)     compare: r21i = 12 layers (5090)
100M tokens (memory Δ shows by eval 1-2; compare r21i at the same token count)."""
import yaml
R = "/home/glenn/projects/neuro/configs/"


def emit(name, n_layers, note):
    c = yaml.safe_load(open(R + "r21i_x1early_150m.yaml"))
    c.update(n_layers=n_layers, attention_layer_positions=list(range(1, n_layers)),
             max_tokens=100_000_000, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r23_configs.py (2026-10-04) from r21i_x1early_150m. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)


emit("screen_r23a_g3w_4L_x1e", 4, "G3 width, 4 layers (depth proxy).")
emit("screen_r23b_g3w_6L_x1e", 6, "G3 width, 6 layers (depth proxy).")
