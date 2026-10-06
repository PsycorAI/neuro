p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
old_start = '<li><b style="color:#2471a3">NEXT SCOPE (Glenn 2026-09-30): CUDA GRAPHS for v4</b>'
i = s.index(old_start); j = s.index("</li>", i) + len("</li>")
new = ('<li><b style="color:#2471a3">CUDA GRAPHS for v4 &mdash; SCOPED 2026-09-30</b> (<code>docs/cuda_graphs_v4_design.md</code>). '
       'Compiled census (1B-probe structure): FPT 7,857 kernels + 80 syncs per fwd+bwd &rarr; lif_exact_ml 5,432 + 0 (&minus;31%; the fused-neuron '
       'estimate was revised UP to +10-40%). Inductor cudagraph trees (<code>max-autotune</code>) are BLOCKED for v4: they do not support grad '
       'accumulation (pytorch#169545; rf-detr disables them for GA&gt;1) and v4 uses GA 9. <b>Plan:</b> Phase 1 = per-spiking-block '
       '<code>torch.cuda.make_graphed_callables</code> (Megatron/TE per-layer style; flag <code>cuda_graph_blocks</code>, needs compile_regional + '
       'lif_exact; MQAR/eval stay un-graphed), ~2-3 days. <b>GATE:</b> after the neuron switch, 1B probe GPU-busy &le;60% &rarr; build; &ge;75% &rarr; '
       'stop (use the batch lever). Phase 2 (whole micro-step + static/padded MQAR, needs a 30M recall screen), Phase 3 (optimizer capture), and a '
       'fused 2048-dim single-head GDN-2 kernel (port target NVlabs/GatedDeltaNet-2, arXiv 2605.22791, same recurrence as ours but H=16/d128) '
       '= DEFERRED behind gates in the doc.</li>')
s = s[:i] + new + s[j:]
open(p, "w").write(s)
print("[OK] to-do updated")
