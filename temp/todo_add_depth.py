p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
anchor = "<li><b>IDEA SWEEP 2026-09-29"
assert s.count(anchor) == 1
new = ('<li><b style="color:#c0392b">DEPTH WARNING: SIGNED SPIKING DOES NOT GAIN FROM A 3RD LAYER (analysis 2026-09-30, Glenn asked)</b>: '
       '2L&rarr;3L val_ppl: tfmr 68.8&rarr;<b>59.38</b> (-14%), unsigned full stack 74.9&rarr;66.72 (-11%), signed 61.24&rarr;63.35/64.65 (<b>+3-5%</b>). '
       'So signed &asymp; 0.89&times; tfmr at 2L but 1.07-1.09&times; at 3L &mdash; the ppl lead may vanish at v4 depth (9L). Signed and depth look like '
       'SUBSTITUTES (signed-vs-unsigned gain: -18% at 2L, only -4% at 3L). 3L also: +17% params (+~38% non-embedding FLOPs), firing ~15% vs 12.7%, '
       'horizon seed-fragile (half-life 2 vs 68 tok), slow start (val_ppl at eval 1: 180-190 vs 153; ratio to 2L closes 1.18&rarr;1.03). Only upside seen: '
       'carry-mode first-16-token ppl 78-79 vs 82-83. [spk, attn, spk] layout; sparsity hinge is on the MEAN over spiking layers (not a bug). '
       '<b>TO-TEST (config-only, 30M, need OK), best-of-both candidates:</b> (A) <b>[spk, attn, attn]</b> = depth from attention, ONE GDN memory '
       '(<code>attention_layer_positions: [1, 2]</code>); (B) 3L signed + <code>depth_scaled_init</code> (slow-start = init?); (C) 3L signed + gap440 w0.15 '
       '(horizon robustness, after r12); (D) matched-compute control: 2L signed with n_neurons ~2800 (is width a better use of the 3rd layer&rsquo;s FLOPs?). '
       '<b>Needs code:</b> (E) memory-free 2nd spiking block (one shared GDN memory, fewer FLOPs, no memory competition); '
       '(F) weight-tied 3rd pass (<code>recurrent_depth</code>) = depth without params. <b>For v4:</b> run a 2/3/4L depth ladder signed vs tfmr before trusting the 0.89&times;.</li>\n  ')
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("[OK] to-do updated")
