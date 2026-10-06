p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "426M tfmr LR check STARTED 13:44 (5090)."
assert s.count(old) == 1
new = old + (" <b>3-LAYER (09-28 14:14): hyb3 [spk,attn,spk]+lev4 = 69.32 | recall 100&times;5 vs 3L tfmr 59.38 &rarr; 1.17&times;</b> (2L: 1.11&times;; the 3L tfmr gains a lot once induction heads form). "
 "hyb3 lacks pre_norm+out_gate (&minus;3% in 2L). IMPLICATION: the gap WIDENS with depth at matched attention budget (tfmr 3 attn layers vs hybrid 1) &rarr; the ~100M/6&ndash;9L honest check MUST sweep the attention ratio (1:1 / 1:2 / 1:3) &mdash; a 7:1 v4 layout is likely too attention-poor. "
 "426M LR check: B12 fits (19.5GB reserved), ~11.4k tok/s &rarr; ~1.5h/arm; full 10B run &asymp; 10 days on the 5090.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
