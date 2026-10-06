p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "hybrid muon_attn_lr 0.02 / 0.01 (5090 streams d/e, after round 2)."
assert s.count(old) == 1
new = old + (" <b style=\"color:green\">ROUND-2 (09-28 07:50):</b> <b>C1 lif_levels 4 (integer spikes 0..4) = 82.85 | recall 100/100/100 = WIN, pure spiking (rate 0.062)</b>; "
 "hybrid s1 82.66 | 100/100/100 &rarr; hybrid CONFIRMED (mean 82.5 vs control 102.9); D1 pre_norm+final_norm 92.0 | 100/100/100 (&minus;7%, borderline); "
 "D2 mem_out_gate 91.9&rarr;92.85 | 100/100/91 (&minus;7%, borderline). Swapped 5080 hyb+learnbeta &rarr; <b>hyb+lev4</b> (done ~10:40). "
 "QUEUED 5090: full stack hybrid+lev4+pre_norm+out_gate (after d), lev4 seed 1 (after e). Running: tern, hyb attnlr 0.01/0.02.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
