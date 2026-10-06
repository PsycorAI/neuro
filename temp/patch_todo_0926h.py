# patch_todo_0926h.py -- induction phase transition + SFT B8 spill + w0.2 recall
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "(3) track recall over training to see when/if induction forms.</li>"
b = ("(3) track recall over training to see when/if induction forms. "
     "<b style=\"color:green\">CONFIRMED (600M-tok AdamW arm):</b> logged loss (LM + 0.5&times;MQAR) flat ~8.0&ndash;8.8 to 197M tok, then <b>drops to 4.98 by 229M</b> and ~4.2&ndash;4.4 after &rarr; classic induction-head phase transition at ~200M tok (30M, 2L). Spiking learns MQAR within the 100M-tok screen &rarr; recall sample-efficiency advantage, NOT a transformer bug. Final recall: <code>logs/tfmrfix_screen.log</code>.</li>")
assert s.count(a) == 1
s = s.replace(a, b)
a2 = "w0.2 arm + alpaca tool_eval + B8 speed check queued (<code>temp/run_sftab2.sh</code>)."
b2 = ("<b>MQAR w0.2: recall 100/100/100/53/28</b> (&asymp; w0.5&rsquo;s 59/25). SFT micro-batch 8 SPILLS on the 5090 (32.1GB, stalled) &rarr; keep B4. Tool evals for w0.2 + alpaca: <code>logs/sftab_tooleval.log</code>.")
assert s.count(a2) == 1
s = s.replace(a2, b2)
open(p, "w", encoding="utf-8").write(s)
print("[OK]")
