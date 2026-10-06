"""mk_gap_r9_configs.py -- 2026-09-29 round 9 (Neuro 45): "memory must carry" screens ON the signed stack
(screen_r5_signed_s0_30m = 61.36; control = signed s0 + screen_r8_signed_s1_30m). Design: docs/contiguous_training_design.md.
!!! NEEDS the staged code (temp/stage_r8: stateful path with MQAR aux, attn_window_*) deployed to src/ first --
live src silently IGNORES attn_window_* and its old stateful path SKIPS the MQAR aux."""
import re
C = "/home/glenn/projects/neuro/configs/"
NL = "\n"
src = open(C + "screen_r5_signed_s0_30m.yaml", encoding="utf-8").read()
body = NL.join(l for l in src.splitlines() if not l.startswith("#")) + NL
def mk(n, what, extra):
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", body, flags=re.M)
    hdr = f"# {n} -- round 9 (2026-09-29). AUTO-GEN temp/mk_gap_r9_configs.py from screen_r5_signed_s0_30m (61.36). Change: {what}." + NL
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b + extra); print("[OK]", n)
ST = "stateful_stream: true          # contiguous streams, GDN state carried across windows (TBPTT)" + NL
WIN = ("attn_window_short: 64          # stochastic attention window (SWAX): 64 w.p. 0.5, else full" + NL +
       "attn_window_p: 0.5" + NL + "attn_window_long_frac: 0.1     # last 10% of steps full window" + NL)
mk("screen_r9a_stateful_s0_30m", "contiguous (stateful) training", ST)
mk("screen_r9b_swastoch_s0_30m", "stochastic attention window 64/full", WIN)
mk("screen_r9c_stateful_swa_s0_30m", "stateful + stochastic attention window", ST + WIN)
