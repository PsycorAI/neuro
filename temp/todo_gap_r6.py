p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "426M AdamW LR untuned (TO-DECIDE: short LR probe)."
assert s.count(old) == 1
new = old.replace("(TO-DECIDE: short LR probe)", "&rarr; <b>Glenn 09-28: RUN the short LR check</b>: configs/screen_tfmr426_lr{3e4,6e4,12e4}.yaml (60M tok each, complete cosine, early-training ranking) via temp/run_5090_seq.sh = screens &rarr; LR check (~3h) &rarr; v3 chain (heal now ~Tue 22:00)")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
