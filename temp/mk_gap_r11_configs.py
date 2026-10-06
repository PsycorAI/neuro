"""mk_gap_r11_configs.py -- 2026-09-30 round 11 (Neuro 45): can firing drop to ~10% / ~5% without losing ppl / recall?
Base = signed + canon_a (screen_r8_signed_canon_s0_30m: 59.74, firing 12.5%). Known: constant lam 2.0 -> 6.7% at +5.4% ppl.
r11a = hinge AT 10% (config-only). r11b/c = late ramp (NEEDS the staged sparsity_ramp flag deployed: temp/stage_spklog)."""
import re
C = "/home/glenn/projects/neuro/configs/"
NL = "\n"
src = open(C + "screen_r8_signed_canon_s0_30m.yaml", encoding="utf-8").read()
body = NL.join(l for l in src.splitlines() if not l.startswith("#")) + NL
def mk(n, what, subs, extra=""):
    b = re.sub(r"^run_name:.*$", f"run_name: {n}", body, flags=re.M)
    for k, v in subs:
        b, cnt = re.subn(rf"^{k}:.*$", f"{k}: {v}", b, flags=re.M); assert cnt == 1, k
    hdr = f"# {n} -- round 11 (2026-09-30). AUTO-GEN temp/mk_gap_r11_configs.py from screen_r8_signed_canon_s0_30m (59.74). Change: {what}." + NL
    open(C + n + ".yaml", "w", encoding="utf-8", newline="").write(hdr + b + extra); print("[OK]", n)
RAMP = lambda final: (f"sparsity_ramp_start_frac: 0.6    # lambda 0.5 until 60% of training, then linear to {final}" + NL +
                      f"sparsity_lambda_final: {final}" + NL)
mk("screen_r11a_hinge10_s0_30m", "hinge at 10% firing (sparsity_target 0.10, lambda 4.0)", [("sparsity_target", "0.10"), ("sparsity_lambda", "4.0")])
mk("screen_r11b_ramp05_s0_30m", "late ramp to ~5% (target 0.05, lambda 0.5 -> 4.0 from 60%)", [], RAMP("4.0"))
mk("screen_r11c_ramp10_s0_30m", "late ramp to ~10% (target 0.10, lambda 0.5 -> 8.0 from 60%)", [("sparsity_target", "0.10")], RAMP("8.0"))
