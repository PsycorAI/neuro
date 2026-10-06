p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
anchor = "<li><b>IDEA SWEEP 2026-09-29"
assert s.count(anchor) == 1
new = ('<li><b>MEMORY HEADS: re-test H=2/4/16 on the CURRENT stack (Glenn asked 2026-09-30; RESEARCH / TO-TEST, low priority)</b>: '
       'the only evidence for 1 head &times; d_mem is the July 30M sweep (<code>neuro_gdn2_dense_chunked</code>: MQAR N16 91/70/11/8% at '
       'H=1/4/8/16, val_ppl fine or better at H&ge;8) &mdash; measured on the OLD stack (FPT neuron, pre-signed, pre-gap-curriculum, pre-leak-fix). '
       'Production GDN models recall well at H=16&times;128 (Gated DeltaNet-2, arXiv 2605.22791), so our cliff may be stack-specific. Payoff if '
       'heads survive: 16&times; smaller state per layer, fused GDN-2 Triton kernels become usable (big speed win, see cuda_graphs_v4_design.md &sect;6 alt). '
       'Cheapest test: 30M signed+canon at H=4 and H=16 (needs the multi-head dense path or fla), judged on recall N16/32/64 + haystack, not ppl.</li>\n  ')
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("[OK] to-do updated")
