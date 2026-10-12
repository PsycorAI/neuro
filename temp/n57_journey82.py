"""n57_journey82.py -- Neuro 57 handoff: append chapter 82 to docs/journey.html (TOC, banner, chapter, version footer)."""
P = "/home/glenn/projects/neuro/docs/journey.html"
s = open(P, encoding="utf-8").read()
assert 'id="demo-build-silent-memory"' not in s
CH = '''
<h2 id="demo-build-silent-memory">82. The demo build starts, a brain takes at 13 layers, and the memory layer goes quiet at 1B</h2>
<p class="date">2026-10-12 (Neuro 57)</p>
<p><a href="#h200-bench-v4-launch">&sect;81</a> left v4 training and five demo features unbuilt. Glenn asked for questions, then a plan, then the build. The plan is <code>docs/demo_v4_plan_2026_10_12.md</code>: ten designs with pass marks fixed in advance, and the SFT done twice on the 5090 (once on the dense-final checkpoint as a fallback and full-scale rehearsal, once on the converted one). His answers changed three things: the model itself announces a tool (&ldquo;I will look that up for you&rdquo;), brains may train weights, and the final SFT runs at home. He chose Python coding and exercise / peak health as the brain domains.</p>
<p>The first night of building gave mostly yes-answers at 13 layers. The new wording is one fixed sentence per tool between the mode word and the call (<code>tool_proto: 2</code>); with all nine tools offered the model picks the right tool 97.5% of the time and gets the arguments exactly right 82.8%. The nine executors are written (<code>scripts/demo/tools_exec.py</code>) with an offline search pack of 222,571 Simple English Wikipedia paragraphs. Answering from a tool result went from 35% with no training to 75.6% with 9,605 teacher-written examples, still short of the 90% mark: search, translate and the two simulated tools are weak. Generation speed turned out to be overhead, not size: 115&ndash;150 ms per token at 189M and at 1.1B alike, with <code>SpikeCode.forward</code> 45% of a forward.</p>
<div class="milestone"><strong>A brain takes.</strong> A rank-32 LoRA adapter on the attention blocks, trained for 45 minutes on 200 invented facts, answered 98.0% of them in wordings it had never seen (base 0.0%), and 0.0% of the control facts it was never taught (<code>brain_mech13</code>). The mechanism works on this architecture; tool retention and energy with the brain on are still unmeasured.</div>
<div class="setback"><strong>Two things did not hold.</strong> Self-knowledge cannot be tested at 13 layers: the probe found 6 of 6,400 science questions known, so &ldquo;search what you do not know&rdquo; collapses to &ldquo;search everything&rdquo;. And no v4 checkpoint could reach home: the pull runs at 0.6 MB/s, four hours for 10 GB, while the instance prunes each checkpoint after 66 minutes. Glenn set up rclone; the Drive backup has been live since 10:05 and a pull from Drive runs three times faster.</div>
<p>The run itself gave the scare of the day. The memory layer&rsquo;s spike rate followed the 13-layer reference for three evals (0.140, 0.107, 0.104) and then kept falling: 0.034, 0.011, down to 0.003 by step 33,000, where the reference had settled near 0.07. The write gates sat at their neutral value. This is what the 150M G3 failure looked like, and I had called the early drop normal a few hours before. Glenn has no money for a second run, so we tested on the instance beside the trainer. The first 64 tokens after a chunk boundary score 54.95 with the memory and 60.77 without (a cold start is 61.12): the layer is sparse, not dead. In-context fact recall is 87.5%, text samples are fluent, and recall across a gap or a chunk is at chance, which the schedule predicts until 50% and 70%. Whether exact recall can be learned at 0.3% spikes is the open question the 55% eval has to answer.</p>
<hr>
<p class="date">Major version: 1.67 (Demo plan + designs written, build started: tool protocol v2 passes at 13L with all 9 tools [.975 / .828]; executors + offline search pack; second pass 35% -> 75.6%; LoRA brain mechanism 98.0% vs 0.0% base and control; self-knowledge untestable at 13L [0.1% known]. v4 at 1B: memory-layer spike rate fell to 0.003 by step 33k, instance tests show it still carries context across chunks [first-64 54.95 vs 60.77 off]; exact cross-chunk recall undecided until ~55%. rclone -> Drive backup live; no checkpoint had reached home before it.)</p>
'''
tail = "\n</body></html>"
assert s.rstrip().endswith("</body></html>")
s = s.rstrip()[:-len("</body></html>")].rstrip() + "\n" + CH + tail + "\n"
toc_old = '<li><a href="#h200-bench-v4-launch">81. The bench, two lost instances, and v4 launched at 36,600 tokens a second</a></li>'
assert s.count(toc_old) == 1
s = s.replace(toc_old, toc_old + '\n  <li><a href="#demo-build-silent-memory">82. The demo build starts, a brain takes at 13 layers, and the memory layer goes quiet at 1B</a></li>')
assert s.count("last updated 2026-10-11.") == 1
s = s.replace("last updated 2026-10-11.", "last updated 2026-10-12.")
open(P, "w", encoding="utf-8").write(s)
print("[OK] journey chapter 82 added")
