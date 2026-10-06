# patch_todo_0926g.py -- add the AdamW transformer arm to the 30M preview entry
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "Resolve (e.g. transformer-standard AdamW LR sweep, 4L) BEFORE the 363M fixed baseline, or its recall comparison is meaningless.</li>"
b = ("<b>UPDATE (AdamW arm, <code>screen_tfmrfix_adamw_s0_30m</code>, lr 1e-3):</b> val_ppl <b>68.8</b> (vs 87.8 on the spiking-tuned NorMuon recipe &rarr; the baseline MUST use a transformer-appropriate optimizer; spiking still &minus;19% at matched params+tokens) but recall still chance (22/16/9/6/3). "
     "Likely NOT a bug: induction heads in small transformers typically form only after ~billions of tokens; 100M-token screens are too short to judge transformer recall, while the Hebbian memory binds key&rarr;value architecturally (a real sample-efficiency point for spiking). "
     "Implications for the 363M fixed baseline: (1) AdamW/transformer-standard LR, not the spiking recipe; (2) report both matched-tokens and matched-compute (tfmr ~6&times; cheaper/token at 30M); (3) track recall over training to see when/if induction forms.</li>")
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
print("[OK]")
