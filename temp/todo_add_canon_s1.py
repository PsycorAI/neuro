p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
anchor = "<li><b>IDEA SWEEP 2026-09-29"
assert s.count(anchor) == 1
new = ('<li><b style="color:green">SIGNED + CANON-A CONFIRMED 2-SEED (r10_signed_canon_s1, 2026-09-30)</b>: val_ppl <b>59.29</b> (s0 59.74; '
       'mean 59.52 vs signed 61.24 = <b>-2.8%</b>, &asymp;0.87&times; the 2L tfmr); recall 100&times;5; haystack N8 gap64/192/440 91/64/35, '
       'N32 81/53/23; half-life 109/64 tok; firing 12.5% (level-1). = the new best 30M stack; r12 (+ gentle gap) builds on it.</li>\n  ')
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("[OK] to-do updated")
