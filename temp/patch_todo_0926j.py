# patch_todo_0926j.py -- round-3 SFT sweep + 426M transformer baseline proposal
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
a = "  <li><b>grammar.py mask-cache BUG (fixed)</b>"
b = ("  <li><b>SFT rehearsal sweep ROUND 3 (RUNNING 2026-09-26 09:12, Glenn: find the best setting between 0.2 and 0.5)</b>: new arms w0.1, w0.3, "
     "hiN-only (N16&ndash;32, w0.2), w0.2 seed 1 (noise) on the 5090 (<code>temp/run_sftw_5090.sh</code>); tool_eval re-run at <b>n=200</b> (free mode) for ALL arms via a shared claim queue on both GPUs "
     "(<code>temp/sftw_lib.sh</code>, <code>logs/tooleval_n200_&lt;arm&gt;.log</code>).</li>\n"
     "  <li><b>PROPOSAL: v3-matched 426M transformer baseline on the 5080</b> (Glenn 2026-09-26: run it during v4 on the 5090?). Config drafted: <code>temp/tfmr426_L17.yaml</code> (17L all-attn d1280, 421.5M, attn_single_residual, AdamW 3e-4, same data/vocab/MQAR/MTP/10B tok as v3). "
     "tok/s + VRAM probe on the 5080 queued (<code>temp/probe_tfmr426_b{8,4}.yaml</code>). Est. ~12&ndash;15 days for 10B tok (to be replaced by the measured number). Gives matched-tokens (vs v3 @10B) and matched-compute (vs v3 ckpts) comparisons.</li>\n"
     + a)
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b, 1))
print("[OK]")
