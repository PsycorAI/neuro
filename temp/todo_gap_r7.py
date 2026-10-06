p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "RUNNING 5090: full stack, lev4 s1, hyb+lev4 at HALF LR (muon 0.0025 / adam 3e-4)."
assert s.count(old) == 1
new = old + (" <b style=\"color:green\">ROUND-4 (09-28 13:15):</b> <b>FULL STACK hybrid+lev4+pre_norm+out_gate = 73.44 | recall 100&times;5 = 1.07&times; tfmr (BEST, 1 seed)</b>; "
 "hyb+lev4 HALF LR 103.1 (worse &rarr; muon 0.005/adam 6e-4 is the optimum: both &times;0.5 and &times;2&ndash;4 hurt); "
 "3-LAYER TRANSFORMER (71.3M) = 59.38 with recall 100&times;5 (induction formed within 100M tok at 3 layers) = the bar for hyb3_lev4 (5080, running).")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
