# patch_todo_0926i.py -- matched-compute result of the 30M transformer preview
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "Final recall: <code>logs/tfmrfix_screen.log</code>.</li>"
b = ("Final (600M tok, 1.36h on the 5080): <b>val_ppl 39.7, recall 100/100/100/100/97</b>.</li>\n"
     "  <li><b style=\"color:#c0392b\">SOBERING &mdash; MATCHED COMPUTE (30M, 2026-09-26)</b>: for LESS wall-clock than spiking&rsquo;s 100M-tok run (~1.7h), the fixed 2L transformer (AdamW, 600M tok) reaches val_ppl <b>39.7 vs spiking 55.8</b> and recall N32/N64 <b>100/97 vs 59/38</b>. "
     "Spiking wins only at matched TOKENS (55.8 vs 68.8 ppl; recall inside 100M tok). Caveats: 30M/2-layer toy; spiking throughput not yet optimized (lif_parallel eager, telemetry syncs); energy-per-token argument is separate and unmeasured here. "
     "<b>Implication for the pitch/Phase 0:</b> the headline must be framed on sample efficiency (+ energy if measured), not raw quality-per-GPU-hour, unless the 363M fixed baseline or throughput work changes this. Glenn to review before any external claim.</li>")
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
print("[OK]")
