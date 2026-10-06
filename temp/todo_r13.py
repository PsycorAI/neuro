p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "(caveat: short runs favour higher LR; 6e-4 is within the usual 400M range). v3 HEAL STARTED 15:24 on the 5090 (resumed @10B)."
assert s.count(old) == 1
new = old + (" <b>ROUND 6 (5080 overnight 09-29):</b> full stack s1 74.92 | recall 100/100/100/100/97 (mean 74.2, CONFIRMED); "
 "canon_a 71.87 | 100&times;5 (&minus;3%, 1 seed, best; haystack still chance); "
 "r6_gap (gap 440, w0.5) 92.55 (+25%, rose late) | recall 88/75/56/41/19 | haystack N8 71/77/70/68, N32 36/48/41/42 = interference FIXED but too costly; "
 "r6_gap_lifexact 121.6 | 50/22/25/9/25 = NEGATIVE. "
 "<b>NEXT (tonight&rsquo;s 5080 queue, in order):</b> screen_r7_gap128_s0_30m (short gaps), screen_r7_gapw015_s0_30m (low weight), canon seed 1, then r5 leftovers (signed, alpha1, alpha4, 3L_2attn, engram32k, mqar025). "
 "IDEAS (captured, TO-TEST unless noted): mqar_gap_prob (mix gap/no-gap MQAR per step; needs a flag); canon_a + gentle gap stack; bigram &lsquo;key taps&rsquo; for the memory key (k from t&minus;1 and t&minus;2, zero-init tap; not built); "
 "honest DeltaProduct retest on the full stack (config-only; was leak-era negative); dense multi-query MQAR supervision (not built); continuous value path from the residual (PARKED: breaks spiking purity; lev4/8 approach it); "
 "1-attn-layer ablation (v3 shape, spiking blocks &rarr; MLP) for the paper.")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
