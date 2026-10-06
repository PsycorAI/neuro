p = "/home/glenn/projects/neuro/docs/journey.html"
s = open(p, encoding="utf-8").read()
toc_anchor = '<li><a href="#width-blocksize-screens-posttrain-ready">'
i = s.index(toc_anchor); j = s.index("</li>", i) + 5
s = s[:j] + '\n  <li><a href="#v3-lands-review-bugs-launch-plan">v3 lands at 10B, a code review finds a residual bug that invalidates our headline comparison, post-training survives an OOM, and a launch plan takes shape</a></li>' + s[j:]
ch = """
<h2 id="v3-lands-review-bugs-launch-plan">66. v3 lands at 10B, a code review finds a residual bug that invalidates our headline comparison, post-training survives an OOM, and a launch plan takes shape</h2>
<p class="date">2026-09-25 (session b)</p>

<p><code>flagship_v3</code> finished cleanly at 10.0B tokens (<code>step_0271268.pt</code>, val_ppl <b>17.57</b>, its best), and v4 got a concrete 1B target shape. Exact CPU builds put A&prime; = d2048 / 9 layers / <code>n_neurons</code> 9728 (4.75&times;d) at 1000.0M. A 30M depth-vs-width screen at matched params (L4 d832 vs the L2 d1024 control from <a href="#width-blocksize-screens-posttrain-ready">the width chapter</a>) showed the deeper model training <b>1.8&times; slower</b> (10.7k vs 18.9k tok/s), which favours staying wide.</p>

<div class="setback"><p><b>The headline comparison was wrong.</b> A careful code review found that <code>HybridAttentionBlock</code> already returns <code>x + attn + ffn</code> and the LM loop adds <code>x</code> again, so every attention layer doubles the residual stream (verified numerically: 44 &rarr; 89). v3 has one attention layer and adapted to it, but the 14-layer matched-transformer baseline grew its stream 2<sup>14</sup>&times;. That handicapped it, so &ldquo;spiking beats transformer by 34%&rdquo; is invalid until re-run. It is also the likely true cause of the <code>final_norm</code> requirement for attention stacks.</p></div>

<p>The review also found that <code>StreamText.batch</code> draws all B windows from one ~37k-token region (correlated, overlapping microbatches), and that the GDN-2 gate telemetry ran every training forward, costing ~14 host syncs and 3 graph breaks per block. We shipped three default-off flags, <code>attn_single_residual</code>, <code>telemetry_eval_only</code> and <code>data_independent_windows</code>, with a parity test showing byte-identical defaults (<code>546968f</code>); the first two are on in the v4 draft. A 30M independent-windows screen is queued. Further throughput candidates (<code>lif_parallel</code>'s O(T&sup2;) scan with per-iteration syncs, Liger fused-linear CE, pinned H2D copies) wait on a first-ever profile. A cleanup moved 494GB of superseded checkpoints, configs and scripts to a trash dir (<code>2fa039e</code>).</p>

<div class="setback"><p><b>Post-training's first launch OOM'd within minutes.</b> SFT pads each batch to its longest example, so T is ragged. <code>_gdn2_dense_chunk</code>'s tail path materialized a (B,C,C,K) fp32 tensor. That was safe when chunks were 64, but at v3's chunk 256 it is ~1.3GB per call. The fix zero-pads the tail to a multiple of the 16-token sub-block (exact no-op rows; parity vs <code>gdn2_sequential</code> ~1e-5 incl. grads, <code>bcf7de7</code>). SFT then ran at ~2.1k tok/s. OPD's 8M-token budget measured out at ~6 days (15 resp tok/s, full-prefix rollouts), so it was cut to 1M (~19h).</p></div>

<p>Research this session went to the backlog: Google's Dream-RSI (it improves an exploration <i>policy</i> by replaying discovery trees and trains no weights; parked as a research-process idea), &ldquo;LMs Need Sleep&rdquo; (consolidate fast-weight memory into slow weights, the closest fit to our Hebbian state), on-policy self-distillation, and Distilled RL. We also set a go-to-market strategy: publish results now, keep the recipe private until v4, open the architecture after. Glenn is founder / Chief Scientist and his son is the intended CEO and public face; <b>brains</b> are to be tested as the unique selling point.</p>

<hr>
<p class="date">Major version: 1.51 (v3 DONE 10.0B, step_0271268, val_ppl 17.57. v4 shape A' d2048/9L/N9728 = 1000.0M; 30M depth screen: L4 1.8x slower at matched params. CODE REVIEW: attention DOUBLE-RESIDUAL [2x+attn+ffn] -> transformer baseline + 34% claim INVALID; correlated-batch sampler; telemetry host-syncs/graph breaks -> flags attn_single_residual / telemetry_eval_only / data_independent_windows [default off, parity-tested, 546968f]. SFT OOM on GDN-2 ragged tail [(B,C,C,K) at chunk 256] -> zero-pad fix bcf7de7. OPD 8M->1M [6d->19h]. 494GB moved to trash. Backlog: Dream-RSI, Sleep consolidation, OPSD, Distilled RL. Launch/funding/team plan in to-do. See neuro_code_review_2026_09_25 + user_glenn_profile_and_venture + HANDOFF_2026_09_25b.)</p>
"""
k = s.rindex("</body>")
s = s[:k] + ch + "\n" + s[k:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
