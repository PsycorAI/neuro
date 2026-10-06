p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
old = "the spiking block is overhead/memory-bound (LIF FPT K=10, GDN scan), "
assert s.count(old) == 1
s = s.replace(old, "the spiking block is launch-bound: the GDN-2 chunk loop (~85% of per-block kernels), NOT the neuron "
                   "(30M screens already use the fused lif_exact_ml kernel, eager; corrected 09-30, docs/fused_neuron_kernel_design.md), ")
anchor = "<li><b>IDEA SWEEP 2026-09-29"
assert s.count(anchor) == 1
new = ('<li><b style="color:#2471a3">FUSED NEURON KERNEL &mdash; SCOPED 2026-09-30: NOTHING NEW TO BUILD</b> (<code>docs/fused_neuron_kernel_design.md</code>). '
       'The fused neuron already exists: <code>lif_exact_ml</code> (1 fwd + 1 bwd Triton kernel; FPT K=10 = 277 kernels + 10 host syncs/call '
       '&rarr; 12 + 0). Fusing FPT REJECTED (non-LIF, loses 136.8 vs 99.3). v4 draft/1B probe still use FPT: whole v4 fwd+bwd 21.6k kernels/80 syncs '
       '&rarr; 19.5k/0 with lif_exact_ml; the neuron is only ~4-10% of launches, <b>the GDN-2 chunk loop is ~85%</b>. '
       '<b>ACTIONS:</b> (A) set <code>lif_exact/lif_levels 4/lif_signed</code> in the v4 config at the 1B re-size; measure FPT vs lif_exact_ml '
       'on <code>temp/probe_1b_b8.yaml</code> when the 5090 is free (pre-registered: &ge;+10% tok/s = speed lever, else adopt for quality only). '
       '(B) triton_op wrapper (removes the 4 compile graph breaks) &rarr; folded into the CUDA-graph scope. (C) kernel tuning only if the 1B profile '
       'shows the neuron &ge;5% of GPU time.</li>\n  '
       '<li><b style="color:#2471a3">NEXT SCOPE (Glenn 2026-09-30): CUDA GRAPHS for v4</b> &mdash; after the fused-neuron work. Targets: the GDN-2 '
       'chunk-loop launches (main win), lif_exact_ml as triton_op (spec in fused_neuron_kernel_design.md &sect;6), the MQAR dynamic-shape compile '
       '(<code>train_gpu.py:538</code>; old cudagraphs+MQAR crash @363M). Alternative/complement: fused single-head large-state GDN-2 Triton kernel. '
       'Status: TO-SCOPE (/new-feature).</li>\n  ')
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("[OK] to-do updated")
