p = "/home/glenn/projects/neuro/docs/journey.html"
s = open(p).read()
toc_old = '  <li><a href="#signed-spikes-memory-horizon">70. Signed spikes beat the transformer on two seeds; the interference failure is a memory that stopped remembering; screens ran at half speed</a></li>\n'
assert s.count(toc_old) == 1
s = s.replace(toc_old, toc_old + '  <li><a href="#gentle-gap-depth-warning-throughput">71. A gentle gap dose fixes long-range recall for free, a third layer exposes a depth problem, and throughput scoping finds the neuron was already fused</a></li>\n')
ch = '''
<h2 id="gentle-gap-depth-warning-throughput">71. A gentle dose of gap training fixes long-range recall for free, a third layer exposes a depth problem &mdash; and throughput scoping finds the fused neuron already built</h2>
<p class="date">2026-09-30 (Neuro 46)</p>

<div class="milestone"><p><b>The gentle gap curriculum works.</b> <a href="#honest-gap-hybrid-levels-interference">&sect;69</a> fixed haystack interference with gap-MQAR at full weight but paid +25% ppl. At <code>mqar_weight 0.15</code> (<code>screen_r7_gapw015</code>, unsigned full stack, gap 440) val_ppl was <b>71.10</b>, slightly better than the ~74.9 control. Haystack recall was 99&ndash;100% at every gap, N8 and N32, and the memory half-life on text went from ~1 token to <b>~3,700</b>. The memory learned to remember without the LM paying for it. <code>r12</code> puts the same dose on the best stack.</p></div>

<p>That best stack is now confirmed. <code>signed+canon</code> seed 1 gave 59.29 against seed 0&rsquo;s 59.74: a two-seed mean of <b>59.52</b>, 2.8% better than signed alone and about 0.87&times; the 2-layer transformer, with recall and horizon intact at 12.5% firing. The sparsity sweep mapped the signed stack&rsquo;s Pareto curve. &lambda; 0.5/1/2/4 gave 61.24/62.73/64.60/67.46 at true firing 12.7/9.3/6.7/4.8%: about +3% ppl per 30% cut in firing, and recall survives at every point.</p>

<div class="setback"><p><b>Depth works against us.</b> Glenn asked whether a third layer is worth some ppl. Going 2L&rarr;3L, the transformer improved 14% (68.8&rarr;59.38) and the unsigned stack 11%, but the signed stack got <em>worse</em> (61.24&rarr;63.35/64.65). So the 0.89&times; lead holds at two layers but becomes ~1.08&times; at three, and v4 plans nine. The signed-vs-unsigned gain also shrinks from 18% to 4%, so signed spikes and depth look like substitutes. The 3L horizon collapse of <a href="#signed-spikes-memory-horizon">&sect;70</a> proved seed-dependent (half-life 2 vs 68 tokens). A third spiking layer also costs +32% FLOPs but <b>1.7&times;</b> wall time. Depth screens <code>r13a</code> ([spk, attn, attn]), <code>r13b</code> (<code>depth_scaled_init</code>) and <code>r13d</code> (2L at 4,096 neurons, param-matched) are queued, to be judged at matched wall-clock.</p></div>

<p>The throughput question then turned into two scoping passes. First, the &ldquo;fused neuron kernel&rdquo; already exists: <code>lif_exact_ml</code>, one Triton forward and one backward kernel, used by every current 30M screen. The 10-iteration FPT still lives only in the v4 draft and the 1B probe. Kernel counts made the point: one FPT call is 277 kernels plus 10 host syncs, the fused neuron 12 plus 0. The compiled census at the 1B-probe structure went from <b>7,857 to 5,432 kernels per fwd+bwd (&minus;31%)</b>, with all 80 syncs gone. Most of what remains is the GDN-2 chunk loop, which also explains the 1.7&times; 3L wall time. We first under-estimated the neuron&rsquo;s share by counting eager launches; the compiled count corrected that, and both docs say so.</p>

<p>Second, CUDA graphs. The old crash (<code>neuro_cudagraphs_incompat_2026_08_09</code>) has a deeper cause than MQAR. Inductor&rsquo;s cudagraph trees keep gradients inside the graph pool, so gradient accumulation across replays is unsupported (pytorch#169545; rf-detr turns graphs off for GA&gt;1), and v4 accumulates over 9 micro-batches. The plan is Megatron/TE-style per-spiking-block <code>torch.cuda.make_graphed_callables</code>, gated on the GPU still being &le;60% busy after the neuron switch. A 1B FPT-vs-<code>lif_exact_ml</code> probe will settle that on the 5090 once Neuro 44&rsquo;s post-training chain exits. Glenn also asked about multi-head memory. The July cliff (recall 91&rarr;11% at 8 heads) was measured on the old stack, so a re-test is logged as a low-priority idea, since 16 heads would unlock fused GDN-2 kernels. Finally, the 5080 scheduler now waits for the GPU to be idle for 5 minutes before every start, because Glenn sometimes plays past midnight.</p>

<p>Pointers: <code>docs/fused_neuron_kernel_design.md</code>, <code>docs/cuda_graphs_v4_design.md</code>, the 2026-09-30 entries in <code>docs/to-do.html</code>, <code>HANDOFF_2026_09_30b.md</code>.</p>

<hr>
<p class="date">Major version: 1.56 (Gentle gap curriculum WORKS: mqar_weight 0.15 + gap 440 = 71.10 [no ppl cost], haystack ~100% at all gaps, half-life ~3700 tok; r12 = signed+canon + it. signed+canon 2-seed 59.52 [0.87x tfmr] = best. Signed sparsity Pareto lambda 0.5/1/2/4 = 61.24/62.73/64.60/67.46 @ 12.7/9.3/6.7/4.8%. DEPTH WARNING: 2L-&gt;3L tfmr -14%, unsigned -11%, signed +3-5% -&gt; lead may vanish at v4 depth; 3L 1.7x wall time; r13a/b/d queued. Throughput: fused neuron already exists [lif_exact_ml]; compiled census 7,857-&gt;5,432 kernels [-31%], 80-&gt;0 syncs; GDN-2 chunk loop dominates. CUDA graphs: trees blocked by grad accumulation -&gt; per-block make_graphed_callables, gated on GPU-busy &le;60% after the neuron switch; 1B probe A/B queued on the 5090. 5080 scheduler waits for a free GPU. See docs/fused_neuron_kernel_design.md + docs/cuda_graphs_v4_design.md + HANDOFF_2026_09_30b.)</p>
'''
end = "\n</body></html>"
assert s.rstrip().endswith("</body></html>")
i = s.rstrip().rfind("</body></html>")
s = s[:i].rstrip("\n") + "\n" + ch + "\n</body></html>\n"
open(p, "w").write(s)
print("[OK] journey ch71")
