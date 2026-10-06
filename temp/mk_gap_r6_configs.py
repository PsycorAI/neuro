"""mk_gap_r6_configs.py -- 2026-09-28 ppl-gap round 6: in-context recall UNDER INTERFERENCE (haystack_recall_eval:
full stack N8 recall 100% -> 8% with a 64-token real-text gap) + Canon-A."""
import re
C = "/home/glenn/projects/neuro/configs/"
NL = "\n"

def mk(src, n, what, extra):
    s = open(C + src + ".yaml", encoding="utf-8").read()
    b = NL.join(l for l in s.splitlines() if not l.startswith("#")) + NL
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", b, flags=re.M)
    hdr = f"# {n} -- ppl-gap round 6 (2026-09-28). AUTO-GEN temp/mk_gap_r6_configs.py from {src}. Change: {what}." + NL
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b + extra)
    print("[OK]", n)

GAP = ("mqar_gap_max: 440          # haystack MQAR: real-text gap g ~ U[0, cur_max] between facts and query" + NL +
       "mqar_gap_ramp_frac: 0.3    # cur_max ramps 1 -> 440 over the first 30% of training (distance curriculum)" + NL +
       "mqar_batch: 16             # half micro-batch for MQAR (longer sequences)" + NL)
mk("screen_gap_hyb_full_s0_30m", "screen_r6_gap_s0_30m", "haystack MQAR distance curriculum on the full stack (73.44)", GAP)
mk("screen_honest_lifexact_s0_30m", "screen_r6_gap_lifexact_s0_30m", "haystack MQAR distance curriculum on lif_exact (99.3)", GAP)
mk("screen_gap_hyb_full_s0_30m", "screen_r6_canon_s0_30m",
   "canon_a 4 (Canon-A residual conv on the spiking block input) on the full stack",
   "canon_a: 4                 # Canon-A (Allen-Zhu Physics of LM 4.1), zero-init residual causal conv" + NL)
