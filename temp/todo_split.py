"""Split docs/to-do.html (2026-09-25): keep current sections, move everything else
VERBATIM into docs/to-do-archive.html (new dated section at the top). Nothing dropped."""
import re, shutil
D = "/home/glenn/projects/neuro/docs/"
src = open(D + "to-do.html", encoding="utf-8").read().split("\n")
head = src[:18]                                  # doctype..<body>
body_end = max(i for i, l in enumerate(src) if "</body>" in l)
body = src[18:body_end]                          # 0-based idx 18 == line 19
def lines(a, b): return src[a - 1:b]             # 1-based inclusive
KEEP = set(range(107, 179))                      # backlog, code review, v4 status, post-v3, toward-v4, housekeeping
archived = [src[i - 1] for i in range(19, body_end + 1) if i not in KEEP and not src[i - 1].strip() == ""]
archived = [l for l in archived if not re.match(r"\s*<h1>", l)]

new = head + ["""<h1>Neuro Project To-Do</h1>
<p class="updated">Last updated: 2026-09-25. Older entries (dated update logs 2026-07 &rarr; 2026-09-21, 500M-era plans, Phase 3, old ideas, older completions) moved verbatim to <a href="to-do-archive.html">to-do-archive.html</a>.</p>

<h2>Currently Running</h2>
<ul>
  <li><b>5090 &mdash; v3 post-training</b> (<code>temp/run_v3_posttrain.sh</code>, <code>logs/v3_posttrain.log</code>): SFT (~2.1k tok/s, ETA ~21:45 2026-09-25) &rarr; ORPO &rarr; OPD (cut to <b>1M</b> response tokens, ~19h) &rarr; evals + frozen heads. Base = <code>flagship_v3_363m/step_0271268.pt</code> (10.0B tok, val_ppl 17.57). SFT first try OOM'd on the GDN-2 ragged-tail bug &mdash; fixed (<code>bcf7de7</code>).</li>
  <li><b>5080 &mdash; 30M screens</b>: independent-windows screen (<code>logs/indepwin_screen.log</code>), then depth screen (L4 d832 vs L2 d1024, <code>logs/depth_screen.log</code>, paused at step 4000). <b>Auto-pause 18:50 2026-09-25</b> (Glenn uses the 5080 ~19:00&ndash;00:00): <code>temp/pause_5080.sh</code>; resume both with <code>temp/resume_5080.sh</code>.</li>
</ul>

<h2>Launch, Funding &amp; Team Plan (added 2026-09-25)</h2>
<p class="updated">Strategy: publish RESULTS now, keep the RECIPE (training code, data mix, hyperparameters, failed-experiment log) private until v4 (1B); then open the architecture code + small weights. Glenn = founder/President/Chief Scientist; son (ML/AI expert) = CEO &amp; public face once the plan + early traction exist. Status: TO-DO unless marked.</p>
<p><b>Phase 0 &mdash; credibility (weeks 0&ndash;3)</b></p>
<ul>
  <li><b>Fixed matched-transformer baseline</b> at 363M (same data/tokens, <code>attn_single_residual: true</code>) &mdash; the old &ldquo;spiking beats transformer 34%&rdquo; claim is INVALID (double-residual bug). #1 experiment for the story. <b>Also retract the 34% / energy numbers in <code>marketing_blurbs.html</code> until re-measured.</b></li>
  <li><b>One crisp, measured win</b>, rigorously: persistent memory across turns/sessions, recall per parameter under split-replay, or spike sparsity/ops-per-token (energy <i>potential</i>, not GPU energy).</li>
  <li><b>Honest comparison vs public small models</b> (SmolLM2-360M, Qwen2.5-0.5B), stated per training token (they saw ~400&times; more).</li>
  <li><b>Licence check</b> before publishing: Llama-3 tokenizer + Llama-3.2-3B teacher (KD/OPD) &rarr; Llama licence naming/attribution rules. Lawyer to confirm.</li>
</ul>
<p><b>Phase 1 &mdash; company + soft launch (weeks 2&ndash;6)</b></p>
<ul>
  <li><b>Incorporate</b> (US, Delaware C-corp is the investor default); assign existing code/IP to the company; founder vesting. Son: check his employer's IP/moonlighting agreement before he contributes.</li>
  <li><b>Provisional patent decision</b> (see 2026-09-25 chat): only if we have a measured, specific method worth claiming (e.g. baking/loading portable brains); must be filed BEFORE any public disclosure; micro-entity USPTO fee $65, attorney-drafted ~$3&ndash;6k.</li>
  <li><b>Tech report</b> (architecture at a high level + results + honest limits) &rarr; arXiv (needs an endorser with cs.LG papers: son if he has them, or a collaborator) + Psycor blog.</li>
  <li><b>Live demo</b>: Hugging Face Space with v3-chat (weights private), showcasing the brain feature. Short demo video. Posts: X, Hacker News, r/LocalLLaMA &mdash; led by the son.</li>
</ul>
<p><b>Phase 2 &mdash; compute without dilution (parallel)</b></p>
<ul>
  <li>Apply: NVIDIA Inception, Microsoft for Startups (Azure credits $1k self-serve &rarr; $150k with VC/accelerator), Google Cloud for Startups (up to $350k, VC-backed tier), small grants (CloudRift $100&ndash;1k). NAIRR only via a US university/nonprofit partner.</li>
  <li><b>Cloud budget ($1&ndash;2k, Glenn)</b>: a single H100 spot was estimated at ~6.3 days / ~$230 for ~20B tokens at v4 scale &mdash; re-estimate for the 1B shape after the throughput profile; likely affordable for v4 + the fixed baseline.</li>
</ul>
<p><b>Phase 3 &mdash; v4 (1B) + open the architecture (months 2&ndash;4)</b></p>
<ul>
  <li>Train v4 (cloud or 5090); release inference code + small open weights under a clear licence (e.g. Liquid-style: free under a revenue threshold); Discord; &ldquo;build a brain&rdquo; tutorials.</li>
</ul>
<p><b>Phase 4 &mdash; team &amp; money (from month 2)</b></p>
<ul>
  <li><b>Team</b>: son joins as CEO; recruit 1&ndash;2 experts by collaborating first (co-author the eval paper with authors of the work we build on). YC co-founder matching as a backup.</li>
  <li><b>Money</b>: pre-seed from technical angels + AI accelerators once demo + report + one credible expert exist.</li>
</ul>
"""]
new += ["<h2>flagship_v4 &mdash; preparation</h2>"] + lines(146, 177)
new += ["<h2>Code review 2026-09-25</h2>"] + lines(122, 130)
new += ["<h2>Research Backlog</h2>"] + lines(107, 120) + lines(132, 145)
new += ["<h2>Housekeeping</h2>", lines(178, 178)[0].replace("<p><b>Housekeeping:</b> ", "<p>")]
new += ["</body>", "</html>", ""]

arch = open(D + "to-do-archive.html", encoding="utf-8").read().split("\n")
ai = max(i for i, l in enumerate(arch) if "<h1>" in l) + 1
block = ["", '<h2>Moved from to-do.html on 2026-09-25 (verbatim)</h2>',
         '<p class="updated">Dated update logs 2026-07 &rarr; 2026-09-21, the old v4 queue, 500M-era plans, Phase 3, &ldquo;Open&rdquo;, New Architectural Ideas and older Recent Completions.</p>'] + archived
shutil.copy(D + "to-do.html", "/home/glenn/projects/neuro/temp/to-do.backup.html")
open(D + "to-do-archive.html", "w", encoding="utf-8").write("\n".join(arch[:ai] + block + arch[ai:]))
open(D + "to-do.html", "w", encoding="utf-8").write("\n".join(new))
print("kept lines:", len(KEEP), "archived lines:", len(archived))
