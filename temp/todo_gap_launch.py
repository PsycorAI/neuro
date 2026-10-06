p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "then stack &rarr; 2 seeds &rarr; one 600M-tok run vs tfmr 39.6."
assert s.count(old) == 1
new = old + (" <b>LAUNCHED 2026-09-27 22:08 (Glenn OK): P0 + A1 + A2 + A3</b> via <code>temp/run_5080_gap_p0a.sh</code> &rarr; <code>logs/gap_screen.log</code> "
             "(5080; auto-kill eval2 &gt; 170; A3 hybrid optimizer auto-picked by <code>temp/gap_pick_hybrid.py</code>; ETA ~07:00 09-28). "
             "FPT s1 SKIPPED (Glenn). 2-SEED lif_exact: val_ppl 99.3 / 106.6 (mean 102.9; <b>seed noise ~7%</b> &rarr; single-seed screen deltas &lt;~7% are noise), "
             "recall 100/100/100/100/94 &amp; 100/100/100/94/94 vs FPT s0 136.8, 100/100/97/62/19 &rarr; lif_exact confirmed.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
print("[OK]")
