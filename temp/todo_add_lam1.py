p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
anchor = "<li><b>IDEA SWEEP 2026-09-29"
assert s.count(anchor) == 1
new = ('<li><b>SIGNED SPARSITY PARETO (r10_signed_lam1, 2026-09-30, 1 seed)</b>: sparsity_lambda 0.5 / 1.0 / 2.0 &rarr; '
       'val_ppl 61.24 (2-seed) / <b>62.73</b> / 64.60, true firing 12.7% / <b>9.3%</b> / 6.7% (all level-1). '
       '~+2.4% ppl per halving of firing. lam1: recall 100%, haystack N8 gap64/192/440 86/60/34, half-life 97/51 tok '
       '(on par with signed). lam4 running; compare r11 hinge/ramp against this curve.</li>\n  ')
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("[OK] to-do updated")
