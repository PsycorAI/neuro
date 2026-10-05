"""mk_r20_mqonce.py -- 2026-10-03: quality A/B for mqar_once_per_step (src/train_gpu.py). The 30M rig runs GA1, where
the flag is a no-op, so both arms use B8 x GA4 (same 16,384 tok/step and the same 32 MQAR samples per step as B32 x GA1).
Base = r19c (relu2 + per-layer threshold 10%, the sparse-MLP candidate). Only batch/GA (+ the flag) differ."""
import yaml
R = "/home/glenn/projects/neuro/configs/"
SRC = "screen_r19c_layer10_s0_30m"
for name, upd, note in [
    ("screen_r20a_ga4ctrl_s0_30m", dict(batch_size=8, grad_accum=4), "B8 x GA4, MQAR every micro-step (control)."),
    ("screen_r20b_ga4mqonce_s0_30m", dict(batch_size=8, grad_accum=4, mqar_once_per_step=True),
     "B8 x GA4, MQAR once per optimizer step (batch x4, weight x4)."),
]:
    c = yaml.safe_load(open(R + SRC + ".yaml")); c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r20_mqonce.py (2026-10-03) from {SRC}. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)
