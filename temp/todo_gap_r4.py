p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "retest CUDA graphs / whole-model compile incl. LIF (was the #1 blocked throughput lever)."
assert s.count(old) == 1
new = old + (" <b style=\"color:green\">ROUND-3 (09-28 10:28):</b> <b>hybrid+lev4 = 75.76 | recall 100&times;5 = 1.11&times; tfmr (BEST)</b>; "
 "hybrid muon_attn_lr 0.01 92.96 / 0.02 100.9 (vs 82.4 at 0.005) &rarr; higher attn LR HURTS; tern 93.1 | 100 (&minus;6%, borderline). "
 "Diag (logs/diag_gap_positions_hyblev4.log): by target freq top-100/100&ndash;1k/1k&ndash;4k/rarest: hybrid 1.06/1.14/1.31/1.83, lev4 1.24/1.24/1.24/1.13, hyb+lev4 1.03/1.06/1.16/1.50 &rarr; "
 "attention fixes FREQUENT tokens, the spiking memory (2 layers) fixes RARE tokens &rarr; <b>NEXT SCREEN (TO-TEST): 3-layer [spk, attn, spk] + lev4 vs a 3-layer transformer (param-matched)</b>. "
 "RUNNING 5090: full stack, lev4 s1, hyb+lev4 at HALF LR (muon 0.0025 / adam 3e-4).")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
