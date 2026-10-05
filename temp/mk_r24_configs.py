"""mk_r24_configs.py -- 2026-10-04 (Glenn OK: focus the 5090 on the v4 memory question; r21d/e/f dropped).
"Both learn" recipes at the G3 shape (150M, 12L), each = r21i (x1 gap MQAR from step 0, full window) + one change:
  r24a  SWAX stochastic window (arXiv 2509.24552): attention uses a 32-token window w.p. 0.5, else full; full for the
        last 10% and at eval. 11 layers x 32 = 352 < 440-token gaps, so short-window steps can't be relayed by attention.
  r24b  stateful segments: contiguous 512-token chunks, memory state carried across chunks (TBPTT, detached), reset
        w.p. 0.1 so cold starts still work. Attention never sees the previous chunk -> the carried memory is the only
        route to it, which pushes later layers to READ the memory (r21c: memory writes but is ignored, delta 0.00).
Control: r21i (same queue)."""
import yaml
R = "/home/glenn/projects/neuro/configs/"


def emit(name, note, **upd):
    c = yaml.safe_load(open(R + "r21i_x1early_150m.yaml"))
    c.update(upd, run_name=name)
    with open(R + name + ".yaml", "w") as f:
        f.write(f"# {name} -- AUTO-GEN temp/mk_r24_configs.py (2026-10-04) from r21i_x1early_150m. {note}\n")
        yaml.safe_dump(c, f, sort_keys=False)
    print("[OK]", name)


emit("r24a_swax32_x1e_150m", "+ stochastic attention window 32/full (SWAX).",
     attn_window_short=32, attn_window_p=0.5, attn_window_long_frac=0.1)
emit("r24b_stateful_x1e_150m", "+ stateful 512-token segments (memory carried, reset_prob 0.1).",
     stateful_stream=True, reset_prob=0.1)
