p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "1-attn-layer ablation (v3 shape, spiking blocks &rarr; MLP) for the paper."
assert s.count(old) == 1
new = old + (" <b style=\"color:green\">09-29 16:47 BREAKTHROUGH CANDIDATE: signed multi-level spikes (lif_signed on the full stack) = 61.36 | recall 100&times;5 | haystack N8 100/88/65/34, N32 100/78/48/19 &rarr; 0.90&times; the 2L tfmr (68.5), interference largely fixed (full stack 100/8/10/15). 1 seed; causality VERIFIED (temp/check_causal_ckpt.py: 0 logit change before p).</b> "
 "CAVEAT: spiking-layer firing 0.263 vs full stack 0.107 (2.5&times; more spike events; sparsity penalty overwhelmed). "
 "Also: 3L full stack 66.72 | 100&times;5 vs 3L tfmr 59.38 (1.12&times;); spkmlp 77.61 (worse, +params) haystack chance. "
 "<b>ROUND 8 CONFIGS (TO-TEST, top priority):</b> screen_r8_signed_s1_30m (confirm), screen_r8_signed_lam2_s0_30m (rate-matched sparsity), screen_r8_signed_canon_s0_30m, screen_r8_signed_3L_s0_30m (vs 3L tfmr 59.38).")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
