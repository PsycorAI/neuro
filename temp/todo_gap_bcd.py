p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "recall 100/100/100/100/94 &amp; 100/100/100/94/94 vs FPT s0 136.8, 100/100/97/62/19 &rarr; lif_exact confirmed."
assert s.count(old) == 1
new = old + (" <b>RESULTS so far (09-28 00:10):</b> P0 tfmr AdamW LR 5e-4 84.6 | 1e-3 68.5 | 2e-3 73.6 &rarr; baseline fair. "
  "A1 NorMuon 0.02/lr 1e-3 KILLED eval2 173.5 (control ~154); A2 AdamW 1e-3 eval1 203 vs ~183 &rarr; higher LR HURTS the spiking model &rarr; "
  "&lsquo;LR is lever #1&rsquo; looks WRONG (pending A2 final). "
  "<b>BUILT 2026-09-28 (Glenn OK, default off, parity-proven byte-identical: temp/parity_gap_flags.py; kernel test temp/test_lif_ml.py; plumbing temp/test_gap_arms.py):</b> "
  "ff_conv, swa_window/swa_heads/swa_dim, pre_norm, mem_out_gate, lif_levels, lif_signed, lif_learn_beta (src/model.py, src/train_gpu.py, src/lif_exact.py lif_exact_ml). "
  "<b>RUNNING on the 5090</b> (3 concurrent streams, temp/run_5090_gap_stream.sh &rarr; logs/gap_5090.log): swa&rarr;lev4 | ffconv&rarr;prenorm&rarr;tern | learnbeta&rarr;outgate "
  "(configs screen_gap_*_s0_30m; kill eval2 &gt; 170).")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
