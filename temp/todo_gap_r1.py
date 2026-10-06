p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "(configs screen_gap_*_s0_30m; kill eval2 &gt; 170)."
assert s.count(old) == 1
new = old + (" <b style=\"color:green\">ROUND-1 RESULTS (09-28 05:05, val_ppl | single-pass recall N16/32/64; control lif_exact 99.3 / s1 106.6 | 100/100/94):</b> "
 "<b>A3 HYBRID [spk,attn] 82.4 | 100/100/97 = WIN (1.20&times; tfmr, was 1.45&times;)</b>, attention on NorMuon 0.005 (untuned for attn); "
 "position ratios 1&ndash;3: 0.91, 4&ndash;15: 1.24, 16&ndash;63: 1.26, 256+: 1.19; by freq top-100 1.06 but rarest bucket 1.83 = remaining gap is RARE tokens (logs/diag_gap_positions_hybrid.log). "
 "A2 AdamW 98.8 BUT recall 0/3% (chance) &rarr; AdamW KILLS recall &rarr; keep NorMuon. "
 "B1 ffconv 99.2 (NULL), B2 swa 106.3 (NULL/worse; recall 97), C2 learnbeta 95.1 (&minus;4%, within 7% seed noise; recall 94). "
 "BUILT muon_attn_lr (src/muon.py, default off). QUEUED: hybrid s1 + hybrid+learnbeta (5080, temp/run_5080_gap_hyb.sh, done by ~10:40); "
 "hybrid muon_attn_lr 0.02 / 0.01 (5090 streams d/e, after round 2).")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
