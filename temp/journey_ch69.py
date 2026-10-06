p = "/home/glenn/projects/neuro/docs/journey.html"
s = open(p, encoding="utf-8", newline="").read()
toc_old = '  <li><a href="#bf16-future-leak-exact-lif">68. A future-token leak hid in the LIF scan; the honest exact neuron wins</a></li>\n'
toc_new = toc_old + '  <li><a href="#honest-gap-hybrid-levels-interference">69. Closing the honest gap: hybrid + multi-level spikes reach 1.07&times;, then an interference test humbles our recall</a></li>\n'
assert s.count(toc_old) == 1
s = s.replace(toc_old, toc_new)
assert s.count("last updated 2026-09-27.") == 1
s = s.replace("last updated 2026-09-27.", "last updated 2026-09-29.")
ch = '''
<h2 id="honest-gap-hybrid-levels-interference">69. Closing the honest gap: a hybrid layer and multi-level spikes bring us to 1.07&times; the transformer &mdash; then an interference test shows our &ldquo;perfect recall&rdquo; only held for immediate questions</h2>
<p class="date">2026-09-27 (evening) &ndash; 09-29</p>

<p>Glenn set the goal after <a href="#bf16-future-leak-exact-lif">&sect;68</a>: take the honest 30M gap (<code>lif_exact</code> 99.3 vs the 2-layer AdamW transformer 68.5, 1.45&times;) to &le;1.0&times;, ideally 0.9&times;, without giving up recall or sparsity. The first move was to measure <em>where</em> the gap lives. <code>temp/diag_gap_positions.py</code> splits validation loss by context position. The ratio was only 1.11&times; with 1&ndash;3 tokens of context, but <b>1.72&times; at positions 4&ndash;15</b> and 1.64&times; at 16&ndash;63, easing to 1.41&times; past 256. With the memory ablated, the curve went flat after position 3: every bit of our context use came from the GDN-2 memory. It is an induction-style store keyed on the previous token, and it has no sense of position. The weakness was precise use of <em>recent</em> context. It also explained why the old <code>pre_conv</code> (09-07) had been our best-perplexity screen while destroying recall: it added local mixing, but on the key/query pointer.</p>

<p>Our first lever was the wrong one. The transformer on our NorMuon recipe had scored 87.8, so we guessed the spiking recipe (tuned under the leak) was simply mis-tuned. It was not. NorMuon 0.02 was auto-killed; halving the LR cost 25% (103.1); AdamW matched perplexity (98.8) but <b>drove recall to chance</b>. The optimizer stays NorMuon 0.005 / AdamW 6e-4. The transformer&rsquo;s own LR (1e-3) was confirmed as its best (5e-4 84.6, 2e-3 73.6), so the 68.5 baseline is fair.</p>

<div class="milestone"><p><b>Two structural wins that stack.</b> We built seven default-off flags in one pass, each parity-proven byte-identical when off (<code>temp/parity_gap_flags.py</code>): <code>ff_conv</code>, <code>swa_window</code>, <code>pre_norm</code>, <code>mem_out_gate</code>, and three neuron options, <code>lif_levels</code> / <code>lif_signed</code> / <code>lif_learn_beta</code> (new <code>lif_exact_ml</code> kernel, 0 spike flips vs float64). We then ran them three at a time on the 5090.
<ul>
<li>A 2-layer hybrid <code>[spk, attn]</code> gave <b>82.5</b> (2 seeds).</li>
<li>Integer multi-level spikes (<code>lif_levels: 4</code>, I-LIF style: integer training, binary-spike inference) gave <b>85.0</b> (2 seeds), with pure spiking and recall 100&times;5.</li>
<li>Stacked with pre-norm and the output gate, the <b>full stack reached 73.4 / 74.9 (seeds 0/1) = 1.07&ndash;1.09&times;</b>, recall 100&times;5.</li>
<li>Canon-A (<code>canon_a</code>, Allen-Zhu&rsquo;s residual short conv) added 71.9 on one seed.</li>
</ul>
The frequency split showed why the two wins stack. Attention fixed common tokens (1.06&times;), multi-level spikes fixed rare ones (1.13&times;).</p></div>

<p>Several levers came up null: <code>ff_conv</code> 99.2, the RoPE window read 106.3, learnable &beta; 95.1, 8 levels 74.3 (vs 4 levels 73.4), and attention LR 0.01/0.02 (93.0/100.9). Depth hurt the comparison. A 3-layer <code>[spk, attn, spk]</code> + lev4 scored 69.3 against a param-matched <b>3-layer transformer at 59.4 (1.17&times;)</b>, because the 3-layer transformer forms induction heads within 100M tokens. Glenn asked whether a 1-attention-layer transformer would be fairer. We kept the 17-layer all-attention model as the 426M baseline, param-matched and standard, and noted a 1-attention-layer model as a future <em>ablation</em>. Its LR check (60M tokens) moved it to 6e-4 (93.4 vs 108.2 at 3e-4). The model runs at 29.9k tok/s, so 10B tokens is ~3.9 days on the 5090.</p>

<div class="setback"><p><b>Our recall only held when the question came straight after the facts.</b> <code>temp/haystack_recall_eval.py</code> puts 0&ndash;440 tokens of real text between the key&ndash;value block and the query. Plain <code>lif_exact</code> held N8 at 100/96/73/40% (gap 0/64/192/440), level with the 3-layer transformer&rsquo;s 100/75/70/44. But <b>the best-perplexity full stack fell to chance at a 64-token gap</b> (100/8/10/15): the hybrid and multi-level levers had bought perplexity by giving up robustness. Our MQAR aux never has a gap. That is the &ldquo;interference wall&rdquo; (arXiv 2609.16183), whose fix is a distance curriculum. <code>mqar_gap_max</code> (haystack MQAR with a ramped real-text gap) did lift N8 to a flat ~70% at every gap, but at full strength it cost 25% perplexity (92.6) and immediate recall. On <code>lif_exact</code> it was plainly negative (121.6). A gentler dose is next, and every headline recall number now carries the haystack column.</p></div>

<p>Two framing pieces came out of the same days. <code>docs/transformer_comparison_scorecard.md</code> lists 25 features in six groups with honest status. Today we measurably win on sparsity, and brains are a unique capability not yet measured. We currently lose on quality, recall under interference, training speed, and post-training stability. <b>Alignment &amp; Safety became its own HIGH-priority group</b> after this month&rsquo;s AI-escape incidents. The first targets are interpretability (monosemanticity of spike neurons vs transformer MLP neurons) and alignment-training response. Those two words enter marketing only after matched, measured wins. Meanwhile v3&rsquo;s heal started on the 5090 (09-28 15:24, resumed at 10B). It is healing steadily: val_ppl 44.8 &rarr; 41.4 by 10.58B. It chains into post-training on the healed base for Glenn&rsquo;s demo.</p>

<hr>
<p class="date">Major version: 1.54 (Honest gap diagnosed: loss ratio by context position 1.11x @1-3, 1.72x @4-15, 1.41x @256+ -&gt; weak precise recent-context use. LR hypothesis wrong [0.02 killed, half LR 103, AdamW 98.8 but recall at chance]. Wins, parity-proven default-off flags: hybrid [spk,attn] 82.5 [2 seeds]; lif_levels 4 85.0 [2 seeds]; full stack hybrid+lev4+pre_norm+mem_out_gate 73.4/74.9 = 1.07-1.09x tfmr; canon_a 71.9 [1 seed]. Nulls: ff_conv, swa read, learn_beta, lev8, attn LR. 3L: 69.3 vs 3L tfmr 59.4 [1.17x]. Haystack recall eval: full stack at chance with a 64-tok gap [lif_exact ~ 3L tfmr]; mqar_gap_max curriculum fixes interference [N8 ~70% flat] but costs +25% ppl at full strength -&gt; gentler dose next. 426M tfmr: 17L kept, LR 6e-4, 29.9k tok/s. v3 heal running [44.8 -&gt; 41.4]. Scorecard with Alignment &amp; Safety HIGH. See neuro_ppl_gap_diagnosis_2026_09_27 + docs/ppl_gap_plan_2026_09_27.md + docs/transformer_comparison_scorecard.md + HANDOFF_2026_09_29.)</p>
'''
tail = "\n\n</body></html>"
assert s.endswith(tail) or s.rstrip().endswith("</body></html>")
idx = s.rfind("</body></html>")
s = s[:idx].rstrip("\n") + "\n" + ch + "\n</body></html>" + s[idx + len("</body></html>"):]
open(p, "w", encoding="utf-8", newline="").write(s)
print("[OK] chapter 69")
