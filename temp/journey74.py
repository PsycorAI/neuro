p = "/home/glenn/projects/neuro/docs/journey.html"
s = open(p).read()
s = s.replace("last updated 2026-10-01.", "last updated 2026-10-03.", 1)
toc72 = '  <li><a href="#depth-via-attention-glass-box-pivot">'
i = s.index(toc72); j = s.index("</li>", i) + len("</li>")
s = s[:j] + ('\n  <li><a href="#v4-candidate-reversal-throughput">73. The v4 candidate beats every 4-layer model; the DSN '
             'costs 9% for circuits 7&times; smaller; the slowness is all in the first layer</a></li>'
             '\n  <li><a href="#speed-plan-gap-shrinks">74. The speed plan: compile fixed, and at 1B the gap shrinks to '
             '1.33&times; once NorMuon is amortized</a></li>') + s[j:]
ch = """
<hr>
<p class="date">Major version: 1.58 (Round 16: c0 43.61 [2 seeds] vs r14d 45.41; DSN costs 9% vs GELU l3 40.13 for 7.6x smaller circuits; t4 48.05 but 2.6-3.6x faster. Scorecard, KG recall eval, 23.15B corpus built. In-context reversal curse in both models. 30M profile: the H0 layer is the cost; whole-model compile collapses c0 to 791 tok/s. See HANDOFF_2026_10_02.)</p>

<h2 id="speed-plan-gap-shrinks">74. The speed plan: a recompile storm fixed, and at 1B the gap to the transformer shrinks to 1.33&times; once the optimizer is amortized</h2>
<p class="date">2026-10-02 (afternoon) &ndash; 10-03 (Neuro 48)</p>

<p>The op profiles first raised a doubt: c0 ran NorMuon unbatched while <code>t4</code> ran fused AdamW. A clean timing split on a free 5090 (a pause-the-queue protocol Glenn asked us to keep) settled it at 30M: NorMuon is ~15% of the gap, the memory layer ~80%. Chunk 256 without chunk checkpointing gives +32% with exact math, and Glenn made it standard from r18 together with <code>telemetry_eval_only</code> and batched NS.</p>

<p>Compile had been failing for a simple reason: every MQAR length recompiled, because GDN-2's chunk loop unrolls on T (530 recompiles in 7 steps), and the DSN k-ramp would have recompiled every step too. <code>mqar_pad_to</code> right-pads MQAR to one length (every layer is causal, so the query logit is unchanged to 1e-5), and the k-WTA moved into a compiler-disabled <code>_kwta</code> helper. Compiled c0 went to <b>60.2k tok/s, 1.95&times;</b> the original.</p>

<div class="milestone"><p><b>G4, the 1B probe.</b> c0-S1 (1.109B: H0 + 12&times;[attention + DSN 4d]) against a param-matched transformer (1.096B) at B8: 5,960 vs 15,251 tok/s, a 2.56&times; gap. But NorMuon's Newton&ndash;Schulz took ~460 of c0's ~690 ms per step. With grad accumulation 8 it amortizes: <b>10,145 vs 13,483 tok/s, a 1.33&times; gap</b>, down from 3.6&times; at the start of the day.</p></div>

<p>Both models keep the GPU ~75&ndash;80% busy at 1B, so CUDA graphs can buy at most ~1.25&times;. Glenn wants them anyway, since they are exact. A prototype (<code>temp/proto_graphs.py</code>) gives 1.22&times; over per-block compile at 30M with bit-identical gradients; the k-ramp still has to be made graph-safe. A B4 compile crash turned out to be an Inductor SplitScan codegen bug on the GDN-2 <code>cumsum</code>; B6 and above compile fine.</p>

<p>Round 17 tuned the DSN on one seed each: k 688 (<code>r17f</code>) 41.42, k 512 + ramp from 40% 41.77, hidden 4096 42.16, ramp from 30% 42.69. The LIF-gated DSN with a 20% sparsity target (<code>r17j</code>) scored only 44.28 but had the best haystack of rounds 16&ndash;17 (N32 at gap 440: 13.3 vs ~7). Seed repeats and an f+g combination run overnight.</p>

<hr>
<p class="date">Major version: 1.59 (Speed plan: 30M split H0 ~80% / NorMuon ~15%; std flags from r18 [chunk256, no chunk ckpt, telemetry_eval_only, batched NS, compile + mqar_pad_to]; compile fix 1.95x; G4 1B gap 2.56x at GA1 -&gt; 1.33x at GA8; GPU busy 75-80%; CUDA graphs prototype 1.22x bit-identical; B4 Inductor SplitScan bug. r17: f 41.42, i 41.77, h 42.16, g 42.69, j 44.28 [best haystack]. See docs/speed_plan_v4_2026_10_02.md + neuro_v4_speed_plan_2026_10_02 + HANDOFF_2026_10_03.)</p>
"""
s = s.replace("\n</body></html>", ch + "\n</body></html>", 1)
open(p, "w").write(s)
print("[OK]")
