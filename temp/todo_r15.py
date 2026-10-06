p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "screen_r8_signed_3L_s0_30m (vs 3L tfmr 59.38)."
assert s.count(old) == 1
new = old + (" <b>v3 HEALED + SFT (09-30):</b> heal done 00:40, val_ppl 44.8 &rarr; 41.05 @11.0B; SFT on the healed base (60M tok, 11.6h): "
 "recall 100/100/97/62/28 (base 100/100/100/81/41); tool_eval n200 constrained mode .960 / tool .959 / args_exact .861 / refuse .962 "
 "(leaky-base W=0.1 was .885/.861/.680/.897) &rarr; sft_v3_363m/step_0003663 = DEMO-ABLE now. abstain_eval selectivity +0.000 (never abstains; ORPO+abstain should fix &mdash; CHECK after ORPO). "
 "ORPO running (14:21 start, ~2.2k tok/s); then OPD (~19h). Neuro 46 queued a ~2h 1B neuron probe A/B after the chain (temp/run_probe_neuron_ab.sh); Neuro 45 owns the 5080 queue.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
