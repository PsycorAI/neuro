p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "Status: TO-RESEARCH (next session, while seed-1 runs finish on the 5080)."
assert s.count(old) == 1, s.count(old)
new = ("Status: RESEARCHED 2026-09-27 &rarr; plan <code>docs/ppl_gap_plan_2026_09_27.md</code>; all screens TO-TEST (need Glenn&rsquo;s OK). "
       "<b>DIAGNOSTIC</b> (<code>temp/diag_gap_positions.py</code>, <code>logs/diag_gap_positions.log</code>): ratio lif_exact/tfmr by context position "
       "1&ndash;3: 1.11 | 4&ndash;15: <b>1.72</b> | 16&ndash;63: <b>1.64</b> | 64&ndash;255: 1.50 | 256+: 1.41 &rarr; the gap is PRECISE RECENT-CONTEXT use, not per-token modeling; "
       "memory-ablated curve is flat after pos 3 (all context use = GDN-2 memory; LIF membrane/Engram add ~nothing past the bigram); memory&rsquo;s long-range gain (16&rarr;256+: &minus;22% vs tfmr &minus;9%) is real. "
       "Best-of-both estimate (tfmr-level local + our long-range gain) &asymp; 0.90&times;. Evidence: pre_conv (local mixing) was the best-ppl 30M screen ever but blurred the k/q pointer &rarr; add local context OFF the memory&rsquo;s pointer path. "
       "Also: tfmr on OUR optimizer = 87.8 (lif_exact only 1.13&times; behind it) &rarr; LR/optimizer is lever #1. "
       "SCREENS (TO-TEST, 30M, single-change vs lif_exact): P0 tfmr AdamW LR 5e-4/2e-3 (fair baseline); A1 NorMuon 0.02 + lr 1e-3; A2 AdamW 1e-3; A3 hybrid [spk,attn] (config-only best-of-both); "
       "B1 ff_conv (identity-init causal conv on spikes feeding W_ff only); B2 SWA read with RoPE (window 64, zero-init gate); C1 integer spikes D=4 (I-LIF); C2 PLIF learnable &beta;; C3 ternary spikes; "
       "D1 pre-norm+final norm; D2 GDN output gate; D3 separate spiking MLP; then stack &rarr; 2 seeds &rarr; one 600M-tok run vs tfmr 39.6.")
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="").write(s)
print("[OK] to-do.html updated")
