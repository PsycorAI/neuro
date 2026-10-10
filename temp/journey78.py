p = "/home/glenn/projects/neuro/docs/journey.html"
s = open(p, encoding="utf-8").read()
ch = """
<h2 id="hippocampus-rc3">78. Why the memory ignored real text, a hippocampus that keeps the surprises, and a spiking model below the attention baseline</h2>
<p class="date">2026-10-06/09 (Neuro 53)</p>

<p>Three cheap diagnostics on <code>r25o</code> explained the setback of <a href="#memory-at-depth-spike-conversion">&sect;77</a>. The memory's write gate sat at 0.83&ndash;0.91 on MQAR key/value tokens and at its 0.10 floor on all ordinary text, real or random: it had learned an "MQAR detector", not a write policy. Keys matched fine (random spans: the right write was the top score 95&ndash;100% of the time), but nothing had been written. Retention was the good news: synthetic facts survived to chunk 8 (T4096). The corpus was not to blame: ~76% of chunk-2 tokens continue a chunk-1 document.</p>

<div class="setback"><p><b>Teaching the gate with synthetic data costs too much.</b> Whole-vocab pairs scattered through text (<code>mqar_xchunk_scatter</code>) opened the gate and tripled real-text copy near the boundary, but cost +17% ppl (+9&ndash;10% at a quarter dose, with word-like phrase keys too). Glenn's idea of accepting the cost and recovering it at the end worked for energy but not here: polish phases at the cosine tail, with a rewarmed LR, with a better-ppl teacher, or with the aux switched off recovered only 2&ndash;3 of the 11 points.</p></div>

<p>An honest energy meter followed: counting a weight read per nonzero input, <code>r25w</code>'s 7.5&times; becomes ~4&times; with on-chip SRAM and 2.3&times; streamed from DRAM, because ~half of all linear inputs stay nonzero (<code>temp/energy_eval_v2.py</code>). Higher thresholds cost +10/+24/+60% ppl for ~4/5/7&times; at 40M conversion tokens; a longer conversion cuts the cost but the learned thresholds drift back down, so <code>spike_theta_frozen</code> was built. Per-site thresholds bought nothing: the cheap sites carry little energy.</p>

<p>The release candidate moved to 1024-token chunks (<code>r26c</code>: chunk-1 40.27 vs 41.74 at 512, in-window fact recall at 200 tokens 62% vs 41%). Glenn asked for a review and literature search; three ideas came out of it. SWAX-style random chunk sizes (<code>attn_chunk_short</code>) let plain LM loss train the memory on real text. A real Llama-3.2-1B teacher in the conversion phase, which already pays for distillation. And HOLA (arXiv 2607.02303): an exact cache of the 64 most surprising earlier tokens beside the delta-rule state, scored by the correction the state actually committed (<code>hola_w</code>, built from the GDN-2 kernel's own <code>U</code>). Glenn wanted the hippocampus in v4 from the start.</p>

<div class="milestone"><p><b>The hippocampus works at 12 layers.</b> At 7L HOLA was neutral (a 7L model cannot even copy in-window). At 12L, <code>r26h</code>: chunk-1 39.30 (RC seeds 40.27/41.33), cross-chunk N32 ~100% at every offset, 97% of 32 facts kept at chunk 8 (RC 69%), real-text copy gain doubled, same tok/s. Random chunks alone (<code>r26f</code>) gave the best real-text copy (+7.7 pts) but forgot faster; both together (<code>r26i</code>) kept everything at 100%. Halving the cross-chunk dose (0.25 from 50%) on top of HOLA gave <code>r26n</code>: 38.09, the best dense model, retention 100/100.</p></div>

<div class="milestone"><p><b>A spiking model below the attention baseline.</b> Converting <code>r26n</code> with the Llama-3.2-1B teacher (<code>r26o</code>): chunk-1 <b>36.55</b> against 38.84 for the attention-only <code>r21b</code>, synthetic recall and retention 100%, energy 5.65&times; compute / 3.19&times; on-chip. This is the v4 recipe (RC3), details in <code>neuro_v4_rc3_recipe_2026_10_09.md</code>.</p></div>

<p>Along the way: a cleanup script deleted two running HOLA runs (keep-lists must skip live dirs), the 5080 cannot hold 12L runs for long, a power cut stopped the 5090, and the 46 GB corpus was split into 2 GiB pieces for a three-day Google Drive upload. Open: real natural-language facts beyond the window are still at chance.</p>

<hr>
<p class="date">Major version: 1.63 (Write gate = MQAR detector; synthetic fixes too costly. Honest energy with weight reads ~3-4x. HOLA hippocampus built: 12L win [r26h]; random chunks best real-text copy; xchunk dose-down -> RC3. Llama-1B KD conversion: r26o chunk-1 36.55 vs attention-only 38.84, memory 100%, 5.65x / 3.19x energy. v4 recipe = RC3.)</p>
"""
assert s.rstrip().endswith("</body></html>")
s = s.rstrip()[: -len("</body></html>")] + ch + "\n</body></html>\n"
toc_old = '  <li><a href="#memory-at-depth-spike-conversion">77. Aligned chunks + memory at mid-depth; every weight multiply turned into spikes</a></li>\n'
assert toc_old in s
s = s.replace(toc_old, toc_old + '  <li><a href="#hippocampus-rc3">78. A hippocampus for surprises; a spiking model below the attention baseline</a></li>\n')
s = s.replace("last updated 2026-10-06.", "last updated 2026-10-09.")
open(p, "w", encoding="utf-8").write(s)
print("[OK] chapter 78 + TOC + date")
