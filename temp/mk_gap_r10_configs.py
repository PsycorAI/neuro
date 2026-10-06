"""mk_gap_r10_configs.py -- 2026-09-30 round 10 (Neuro 45): fill-ins for the 5080 afternoon (~1h each with the MADV fix).
(1) signed 3L seed 1: the 3L memory-horizon collapse (half-life ~2 tok, haystack chance, 64.65) -- bistable or systematic?
(2)(3) sparsity sweep around lam2 to map the signed ppl-vs-spike-rate Pareto curve (lam 0.5 = 61.36 @ 0.263; 2.0 = 64.60 @ 0.144)."""
import re
C = "/home/glenn/projects/neuro/configs/"
NL = "\n"
def mk(src_name, n, what, subs):
    src = open(C + src_name + ".yaml", encoding="utf-8").read()
    b = NL.join(l for l in src.splitlines() if not l.startswith("#")) + NL
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", b, flags=re.M)
    for k, v in subs:
        b, cnt = re.subn(rf"^{k}:.*$", f"{k}: {v}", b, flags=re.M); assert cnt == 1, k
    hdr = f"# {n} -- round 10 (2026-09-30). AUTO-GEN temp/mk_gap_r10_configs.py from {src_name}. Change: {what}." + NL
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b); print("[OK]", n)
mk("screen_r8_signed_3L_s0_30m", "screen_r10_signed_3L_s1_30m", "seed 1 of signed 3L (horizon collapse bistable?)", [("seed", "1")])
mk("screen_r5_signed_s0_30m", "screen_r10_signed_lam1_s0_30m", "sparsity_lambda 0.5 -> 1.0 (Pareto sweep)", [("sparsity_lambda", "1.0")])
mk("screen_r5_signed_s0_30m", "screen_r10_signed_lam4_s0_30m", "sparsity_lambda 0.5 -> 4.0 (Pareto sweep)", [("sparsity_lambda", "4.0")])
mk("screen_r8_signed_canon_s0_30m", "screen_r10_signed_canon_s1_30m", "seed 1 of signed + canon_a (59.74, best)", [("seed", "1")])
