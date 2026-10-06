p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "repetition/looping (#16), noisy-input robustness (#17), jailbreak incl. state poisoning (#18), continual learning via brains (#23)."
assert s.count(old) == 1
new = old + (" <b style=\"color:#c0392b\">ALIGNMENT &amp; SAFETY = HIGH (Glenn 09-28, after this month&rsquo;s AI escape/sandbox incidents):</b> new scorecard group D &mdash; "
 "#19 interpretability (auto-interp monosemanticity of spike neurons vs tfmr MLP neurons; spike-clamp steering), #20 alignment-training response (why-lessons, matched recipe), "
 "#24 memory auditability/erase (NEW), #15 honesty/calibration, #18 jailbreak incl. state poisoning, #25 instruction/context isolation (prompt injection; NEW). "
 "Goal: DECISIVE, matched, pre-registered wins before &lsquo;alignment&rsquo;/&lsquo;interpretability&rsquo; go into marketing. TO-SCOPE via /new-feature.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
