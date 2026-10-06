p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
anchor = "<li><b>IDEA SWEEP 2026-09-29"
assert s.count(anchor) == 1
new = ('<li><b>SIGNED 3L HORIZON COLLAPSE IS SEED-DEPENDENT (r10_signed_3L_s1, 2026-09-30)</b>: val_ppl 63.35 (s0 64.65); '
       'memory half-life on text 68/36 tok (s0: 2/2); haystack N8 gap64/192/440 80/57/35 (s0 22/18/15), N32 61/31/16. '
       'Firing 14.6/15.8% (level-1 almost only). The horizon is a fragile, seed-dependent equilibrium &rarr; the gap curriculum '
       '(r12) is the robust fix, not depth. 3L still does not beat 2L signed (61.24) on ppl.</li>\n  ')
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("[OK] to-do updated")
