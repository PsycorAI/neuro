p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "repetition/looping (#16), noisy-input robustness (#17), jailbreak incl. state poisoning (#18), continual learning via brains (#23)."
assert s.count(old) == 1
new = old + (" <b>09-28 16:42:</b> r5 lev8 (on full stack) = 74.29 | recall 100&times;5 vs full 73.44 &rarr; NULL, keep 4 levels. "
 "<b>426M tfmr LR check (60M tok): 3e-4 108.15 | 6e-4 93.43 (&minus;14%) | 1.2e-3 killed at eval2 187 (generic 30M kill rule; worse early) &rarr; baseline lr set to 6e-4</b> "
 "(caveat: short runs favour higher LR; 6e-4 is within the usual 400M range). v3 HEAL STARTED 15:24 on the 5090 (resumed @10B).")
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s); print("[OK]")
c = "/home/glenn/projects/neuro/configs/baseline_tfmr426_L17.yaml"
t = open(c, encoding="utf-8", newline="").read()
assert t.count("\nlr: 0.0003\n") == 1
t = t.replace("\nlr: 0.0003\n", "\nlr: 0.0006                  # 2026-09-28 LR check (60M tok): 3e-4 108.2, 6e-4 93.4, 1.2e-3 worse early\n")
open(c, "w", encoding="utf-8", newline="").write(t); print("[OK] tfmr lr -> 6e-4")
