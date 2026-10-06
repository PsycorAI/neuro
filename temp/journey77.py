"""journey77.py -- 2026-10-06: chapter 77 (Neuro 51) + TOC + banner."""
P = "/home/glenn/projects/neuro/docs/journey.html"
s = open(P).read()
CH = """
<h2 id="memory-at-depth-spike-conversion">77. Attention in aligned chunks, the hippocampus moved to mid-depth, and every weight multiply turned into spikes</h2>
<p class="date">2026-10-05/06 (Neuro 51)</p>

<p>The two overnight recipes of <a href="#h200-memory-ignored-demo">&sect;76</a> both failed at 150M: <code>r24a</code> (SWAX window 32/full) ended at 45.09 with memory-&Delta; +0.3, <code>r24b</code> (stateful segments, detached state) at 45.79 with &Delta; +0.09. The late x1 dose (<code>r21b</code>, 37.59, +3.1%, in-window haystack 100%) became the fallback.</p>

<p>The new idea was <code>attn_chunk</code>: attention only inside <em>aligned</em> 512-token chunks, the same boundaries in every layer, with the memory running across the whole 1024-token sequence at full gradient. A sliding window leaks through L&times;W relay; aligned chunks cannot be crossed by any layer, so the memory is the only route to the previous chunk. New evals measure what the demo needs: per-chunk ppl with memory on/off, haystack with the key block in chunk A and the query in chunk B, NL facts, and real-text copy (<code>temp/run_r25_evals.sh</code>).</p>
<ul>
<li>At 6L the memory is used at once (chunk-2 &Delta; +10), but in-window recall collapses: when the memory is used at shallow depth it takes recall from attention and does it worse.</li>
<li>At 12L (<code>r25a</code>) the memory hurts at eval 1 (&Delta; &minus;6) and is then ignored, chunks or not. A late read path (<code>mem_read_layers</code>) and late phases on top of <code>r21b</code> make the memory matter at no ppl cost, but cross-chunk recall stays at chance or plateaus ~60% (<code>r25f/h/i/k</code>); a memory block inserted late stays silent (<code>r25l/m</code>).</li>
</ul>
<div class="milestone"><p><b>The memory belongs at mid-depth.</b> <code>r25j</code> (memory block at depth 6, chunked, from scratch): cross-chunk haystack 100% for N8 and N32 (memory off: chance), in-window 100%, but +22% ppl. Adding it instead of replacing an attention layer, and dropping the layer-0 block, gave <code>r25o</code>: 12 attention layers + one spiking memory at depth 6, chunk-1 41.74 (+7.5% vs <code>r21b</code>), cross-chunk ~100/84&ndash;100%. The 7-layer analog <code>p25o</code> costs only +0.8%. Contextual keys at depth are what make beyond-window recall work.</p></div>

<div class="setback"><p><b>Real text is still not remembered beyond the window.</b> The memory recalls synthetic key&ndash;value pairs across chunks, but a repeated 32-token real-text span is copied at 30.0% vs 28.2% with memory off (<code>temp/xcopy_eval.py</code>). Natural-language fact episodes (<code>nlfact_every</code>) and a cross-chunk copy augmentation (<code>xcopy_frac</code>, <code>r25v</code>: no gain, +9% ppl) did not fix it, nor does a layer-0 block. The demo will use a key&ndash;value memory game for now.</p></div>

<p>Glenn then asked to raise the energy savings while keeping the spiking hippocampus. Measured honestly (<code>temp/energy_eval.py</code>, 45 nm MAC 4.6 pJ / AC 0.9 pJ), <code>r25o</code> is only 1.39&times; cheaper than the same model run densely: every linear layer reads a dense input, and the LM head alone is a quarter of the cost. Following SpikingBrain and SpikingMamba we built <code>spike_code</code> (<code>src/spike_code.py</code>): every linear input becomes a signed integer spike count with a learned threshold, so a weight multiply becomes accumulates for spikes only, applied as a late conversion phase seeded from the trained checkpoint (<code>temp/mk_spike_seed.py</code>).</p>
<div class="milestone"><p><b>Conversion is free in ppl.</b> <code>r25t</code> (8 levels, 80M tokens of conversion): chunk-1 41.67 vs dense 41.74, memory intact, <b>5.0&times;</b> energy. Coding q and v too (spike-driven attention) adds +1.4% zero-shot and lifts it to 6.2&times;; raising thresholds 1.5&times; with self-distillation gave 7.4&times; at 6L (+10%). Ternary spikes (+89%) were dropped. A self-distillation run diverged because Liger's fused cross-entropy overwrites the logits in place; the KD term is now computed before the CE.</p></div>

<p>Glenn's decisions: v4 may cost up to ~8% ppl against the attention-only baseline in exchange for beyond-window memory and lower energy (2&ndash;8% per idea is fine if it buys something), and the layer-0 block can go. Running at the close: <code>r25w</code> (q/v coding + KD + 1.5&times; thresholds) and the factorised LM head <code>r25y</code> (rank 256/512, SVD-initialised during conversion). Details: <code>docs/memory_beyond_window_2026_10_05.md</code>, <code>docs/energy_plan_2026_10_06.md</code>.</p>

<hr>
<p class="date">Major version: 1.62 (Memory at depth solved for synthetic recall: aligned chunked attention + one spiking memory block at mid-depth [r25o: +7.5% ppl, cross-chunk haystack ~100%]; late phases / insertion fail; real-text recall beyond the window still open. Spike conversion of every linear input [r25t: no ppl cost, 5.0x energy; q/v coding 6.2x; KD fixed]. Glenn: ppl budget ~8%, drop H0.)</p>
"""
end = "\n</body></html>"
assert s.rstrip().endswith("</body></html>") and "memory-at-depth-spike-conversion" not in s
i = s.rstrip().rfind("</body></html>")
s = s[:i].rstrip("\n") + "\n" + CH + "\n</body></html>\n"
toc_old = '  <li><a href="#h200-memory-ignored-demo">76. The H200 is twice as fast as needed; the memory writes but nobody reads it; demo page stage 1</a></li>\n'
assert toc_old in s
s = s.replace(toc_old, toc_old + '  <li><a href="#memory-at-depth-spike-conversion">77. Aligned chunks + memory at mid-depth; every weight multiply turned into spikes</a></li>\n', 1)
s = s.replace("Live document — last updated 2026-10-04.", "Live document — last updated 2026-10-06.", 1)
open(P, "w").write(s)
print("[OK] journey ch 77")
