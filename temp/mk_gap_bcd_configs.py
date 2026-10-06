"""mk_gap_bcd_configs.py -- 2026-09-28: ppl-gap plan B/C/D single-change arms (docs/ppl_gap_plan_2026_09_27.md).
Each = the lif_exact control (screen_honest_lifexact_s0_30m, 99.3 / s1 106.6) + ONE change."""
import re
C = "/home/glenn/projects/neuro/configs/"
src = open(C + "screen_honest_lifexact_s0_30m.yaml", encoding="utf-8").read()
body = "\n".join(l for l in src.splitlines() if not l.startswith("#")) + "\n"
ARMS = {
  "ffconv":    ("B1", "ff_conv 4: causal depthwise conv (zero-init residual) on spikes feeding W_ff only", "ff_conv: 4\n"),
  "swa":       ("B2", "swa_window 64: RoPE windowed-attention read (4 heads x 64, v = memory v), zero-init gate", "swa_window: 64\nswa_heads: 4\nswa_dim: 64\n"),
  "learnbeta": ("C2", "lif_learn_beta: per-neuron learnable membrane decay (init 0.9)", "lif_learn_beta: true\n"),
  "lev4":      ("C1", "lif_levels 4: integer multi-level spikes 0..4 (I-LIF)", "lif_levels: 4\n"),
  "tern":      ("C3", "lif_signed: ternary spikes -1/0/+1 (rate = mean |s|)", "lif_signed: true\n"),
  "prenorm":   ("D1", "pre_norm + final_norm: RMSNorm before to_current, RMSNorm before head", "pre_norm: true\nfinal_norm: true\n"),
  "outgate":   ("D2", "mem_out_gate 16: memory-read output gate 2*sigmoid(low-rank), identity at init", "mem_out_gate: 16\n"),
}
for tag, (pid, what, extra) in ARMS.items():
    n = f"screen_gap_{tag}_s0_30m"
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", body, flags=re.M)
    hdr = (f"# {n} -- ppl-gap plan {pid} (2026-09-28, docs/ppl_gap_plan_2026_09_27.md). AUTO-GEN temp/mk_gap_bcd_configs.py\n"
           f"# from screen_honest_lifexact_s0_30m (control 99.3 / s1 106.6). ONE change: {what}.\n")
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b + f"# --- {pid} ---\n" + extra)
    print("[OK]", n)
