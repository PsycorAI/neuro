"""mk_gap_r8_configs.py -- 2026-09-29 round 8: follow up signed multi-level spikes (screen_r5_signed_s0_30m = 61.36,
recall 100x5, haystack N8 100/88/65/34; causality verified). Caveat: spiking-layer rate 0.263 vs full stack 0.107."""
import re
C = "/home/glenn/projects/neuro/configs/"
NL = "\n"
src = open(C + "screen_r5_signed_s0_30m.yaml", encoding="utf-8").read()
body = NL.join(l for l in src.splitlines() if not l.startswith("#")) + NL
def mk(n, what, subs=(), extra=""):
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", body, flags=re.M)
    for k, v in subs:
        b, cnt = re.subn(rf"^{k}:.*$", f"{k}: {v}", b, flags=re.M)
        if not cnt: b += f"{k}: {v}" + NL
    hdr = f"# {n} -- round 8 (2026-09-29). AUTO-GEN temp/mk_gap_r8_configs.py from screen_r5_signed_s0_30m (61.36). Change: {what}." + NL
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b + extra); print("[OK]", n)
mk("screen_r8_signed_s1_30m", "seed 1 (confirm)", [("seed", "1")])
mk("screen_r8_signed_lam2_s0_30m", "sparsity_lambda 0.5 -> 2.0 (pull firing back toward the full stack's rate; honest sparsity/energy)", [("sparsity_lambda", "2.0")])
mk("screen_r8_signed_canon_s0_30m", "+ canon_a 4 (canon was -3% on the full stack)", [("canon_a", "4")])
mk("screen_r8_signed_3L_s0_30m", "3 layers [spk, attn, spk] (vs 3L tfmr 59.38, 3L full stack 66.72)", [("n_layers", "3")])
