p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p, encoding="utf-8", newline="").read()
old = "Running: tern, hyb attnlr 0.01/0.02."
assert s.count(old) == 1
new = old + (" <b>426M TRANSFORMER BASELINE QUEUED</b> (Glenn 09-28: GPU gaps, screens first): configs/baseline_tfmr426_L17.yaml (421.5M, B12xGA6, ckpt_every 1000, keep_last 3, auto-resume), "
 "temp/run_5090_tfmr426.sh waits until no run_5090_gap_stream is running, then starts on the 5090 (logs/baseline_tfmr426_L17.log). "
 "<b>TO-DO (v4 design, idea capture):</b> the 30M wins (hybrid 1:1, lev4, pre_norm, out_gate) are 2-layer results; v4 choices made on LEAKY val_ppl (7:1 attn ratio, width, depth, sparsity target 0.15) need an HONEST mid-scale (~90&ndash;150M, 6&ndash;9L) confirmation of the stacked recipe + attention ratio sweep (1:1 / 3:1 / 7:1) before the 1B shape is frozen. "
 "Also: lif_exact has no data-dependent loop &rarr; retest CUDA graphs / whole-model compile incl. LIF (was the #1 blocked throughput lever).")
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new)); print("[OK]")
