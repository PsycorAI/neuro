# patch_todo_0926e.py -- 1B probe B=8/B=4 results
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "  <li><b>v4 lever TO-TEST: <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code></b>"
b = ("  <li><b style=\"color:#c0392b\">1B A&prime; PROBE RESULT (2026-09-26, 5090, 72 seqs/step, max-autotune)</b>: "
     "B=4 ~2.2k tok/s @23.9GB reserved; <b>B=8 ~4.0k tok/s @28.2GB</b> (sys peak 29.5GB, under the cliff); B=12 2.1k (spill, 33.7GB). "
     "Overhead-bound (B4&rarr;B8 nearly 2&times;). At 4.0k tok/s: <b>16B tok = ~46 days</b>; 30 days = ~10.4B tok. Target needs &ge;6.2k tok/s (1.55&times;) &rarr; levers MUST stack. "
     "Next: B=12 + expandable_segments probe queued (<code>temp/run_1b_probe_expseg.sh</code>).</li>\n" + a)
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b, 1))
print("[OK]")
