"""Append journey chapter 75 (2026-10-03/04) + TOC entry + date line."""
P = "/home/glenn/projects/neuro/docs/journey.html"
s = open(P).read()
ch = """<h2 id="sparse-mlp-hippocampus-at-depth">75. A sparse MLP beats the spiking one on every axis &mdash; then at 150M the hippocampus goes silent, because attention can already reach everything</h2>
<p class="date">2026-10-03 &ndash; 10-04 (Neuro 49)</p>

<p>The overnight seed repeats settled the DSN question of <a href="#speed-plan-gap-shrinks">&sect;74</a>: <code>r17k</code> (k 688 + ramp from 30%) averaged 41.17 over three seeds, +2.2% against the GELU MLP of <code>l3</code> (40.27), with haystack ~25% weaker in every DSN variant. GELU won on quality but its circuits were ~5&times; larger. Glenn asked whether GELU could get the three factors (sparsity, interpretability, energy) instead.</p>

<p>Round 19 swapped GELU for relu&sup2; (Primer) with an optional <b>homeostatic threshold</b> (<code>SparseAct</code>, <code>attn_ffn_homeo</code>): a neuron fires only above a threshold that is not trained by gradient but nudged after each step so the firing rate tracks a target that ramps from 1.0 to the goal between 60% and 90% of training. LIF membranes stay in the hippocampal layer only.</p>

<div class="milestone"><p><b>Sparsity turns out to be free.</b> Three seeds each: per-layer threshold at 10% active <b>39.88</b>, per-neuron 10% 40.09, 5% <b>40.35</b>, all at or below GELU's 40.27, with GELU-level haystack. At 5% the LM circuits are 33.7k entries against GELU's 330.6k (9.8&times; smaller) and the units are the most selective of any arm. It also trains 9% faster than DSN at 1B. Glenn chose 5%: interpretability matters more to him than 1% perplexity.</p></div>

<p>Speed work in parallel. <code>compile_regional</code> had compiled nothing (the LM calls <code>forward_sequence</code>, not <code>forward</code>); fixed, it is still slower than whole-model compile. The MQAR aux pass turned out to be ~32% of the 1B step (16.7k tok/s with it off vs 11.3k on). <code>mqar_once_per_step</code> is quality-neutral at 30M (39.81 vs 39.96) but at 1B it exceeds the 5090's 32 GB; it is meant for the H200.</p>

<div class="setback"><p><b>G3: at 150M the hippocampus is never used.</b> At 12 layers and 1B tokens the candidate (H0 + 11&times;[attention + sparse MLP]) scored 27.26 against the fair transformer's 26.71 and ran ~1.8&times; slower. Worse, its memory-&Delta; was &le;0 from the first eval, the write gate sat 97% at its floor, and the spike rate fell from 0.28 to 0.004, while recall stayed 100%. At 30M, recall had been 100% memory. Haystack was equally poor in both models (~20% at N8).</p></div>

<p>Two things explain it. First, haystack is a training problem: the late gap dose <code>x1</code> gives 100% at every gap at 30M, with the sparse MLP too (<code>r22a</code>, +1.7%). Second, the memory only works when attention cannot reach. A fixed 128-token window (<code>attn_window_fixed</code>) gives 100% beyond-window recall through memory at 30M (<code>r22b</code>, +2.9%). But at 150M it does nothing for the memory, because 11 stacked windows reach ~11&times;128 tokens, more than the 512-token context. SWAX (arXiv 2509.24552) reports the same: memory learns only when attention is short-sighted, with 1:1 memory:attention layers and 16k training sequences. Round 21 now tests a 32-token window and the gap dose from step 0.</p>

<p>The deadline is now concrete: Glenn wants to demo v4 to his son by <b>October 19</b>. Pretraining will run on an interruptible Vast H200 (~$0.44/h), so the config has to freeze by about October 8.</p>

<hr>
<p class="date">Major version: 1.60 (r17 seeds: DSN r17k 41.17 vs GELU 40.27. r19 sparse MLP [relu2 + homeostatic threshold] 39.88 at 10% / 40.35 at 5% = GELU quality, circuits 8-10x smaller; Glenn picks 5%. Regional compile fixed but not a lever; MQAR aux = 32% of the 1B step; mqar_once_per_step quality-neutral. G3 150M: hippocampus unused at depth [beta collapse, spikes 0.004]; x1 = haystack 100% at 30M; window 128 forces memory at 30M but not at 12 layers [reach ~L x W]; SWAX. Demo Oct 19, Vast H200. See neuro_g3_hippocampus_unused_2026_10_04 + HANDOFF_2026_10_04.)</p>

"""
assert s.count("</body></html>") == 1
s = s.replace("</body></html>", ch + "</body></html>")
toc_old = '  <li><a href="#speed-plan-gap-shrinks">'
i = s.index(toc_old); j = s.index("\n", i)
s = s[:j + 1] + '  <li><a href="#sparse-mlp-hippocampus-at-depth">75. A sparse MLP beats the spiking one on every axis; at 150M the hippocampus goes silent</a></li>\n' + s[j + 1:]
s = s.replace("last updated 2026-10-03.", "last updated 2026-10-04.", 1)
open(P, "w").write(s)
print("[OK] chapter 75")
