p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
old = "lam4 running; compare r11 hinge/ramp against this curve.</li>"
assert s.count(old) == 1
new = ("<b>lam4 (2026-09-30): val_ppl 67.46, firing 4.8%</b>, recall 100%, haystack N8 84/57/30, half-life 87/42. "
       "Full curve (lambda 0.5/1/2/4): 61.24/62.73/64.60/67.46 @ 12.7/9.3/6.7/4.8% firing &rarr; smooth, "
       "~+3% ppl per 30% cut in firing; recall/haystack survive at all points. Compare r11 hinge/ramp against this curve.</li>")
s = s.replace(old, new)
open(p, "w").write(s)
print("[OK] to-do updated")
