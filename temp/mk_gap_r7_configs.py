"""mk_gap_r7_configs.py -- 2026-09-29 ppl-gap round 7: GENTLER haystack-MQAR dose. r6_gap (gap 440, weight 0.5) fixed
interference (haystack N8 ~70% flat) but cost +25% val_ppl (92.6 vs 73.4) and immediate recall. Control = full stack 73.44/74.92."""
import re
C = "/home/glenn/projects/neuro/configs/"
NL = "\n"
src = open(C + "screen_gap_hyb_full_s0_30m.yaml", encoding="utf-8").read()
body = NL.join(l for l in src.splitlines() if not l.startswith("#")) + NL
def mk(n, what, extra, subs=()):
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", body, flags=re.M)
    for k, v in subs:
        b, cnt = re.subn(rf"^{k}:.*$", f"{k}: {v}", b, flags=re.M); assert cnt == 1, k
    hdr = f"# {n} -- ppl-gap round 7 (2026-09-29). AUTO-GEN temp/mk_gap_r7_configs.py from screen_gap_hyb_full_s0_30m. Change: {what}." + NL
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b + extra); print("[OK]", n)
G = lambda gmax: (f"mqar_gap_max: {gmax}" + NL + "mqar_gap_ramp_frac: 0.3" + NL + "mqar_batch: 16" + NL)
mk("screen_r7_gap128_s0_30m", "haystack MQAR with SHORT gaps (max 128) at weight 0.5", G(128))
mk("screen_r7_gapw015_s0_30m", "haystack MQAR gap 440 at LOWER weight (mqar_weight 0.15)", G(440), [("mqar_weight", "0.15")])
