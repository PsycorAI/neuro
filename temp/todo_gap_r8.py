p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "= the bar for hyb3_lev4 (5080, running)."
assert s.count(old) == 1
new = old + " <b>lev4 seed 1 = 87.10 | recall 100&times;5 &rarr; lif_levels 4 CONFIRMED 2-seed (mean 85.0 vs control 102.9, &minus;17%).</b> 426M tfmr LR check STARTED 13:44 (5090)."
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
