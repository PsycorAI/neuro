# journey_ch67.py -- 2026-09-26: backfill TOC drift, add chapter 67 + version footer 1.52, bump banner.
import re, html
p = "docs/journey.html"
s = open(p, encoding="utf-8").read()

# ---- 1) TOC backfill: every <h2 id> must have a TOC <li>, in document order
toc_start = s.index('<div class="toc">'); toc_end = s.index("</ol>", toc_start)
toc = s[toc_start:toc_end]
heads = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', s)
missing = [(i, t) for i, t in heads if f'href="#{i}"' not in toc]
def short(t):
    t = re.sub(r"<[^>]+>", "", t)
    return t.split(" &mdash; ")[0] if len(t) > 110 else t
add = "".join(f'  <li><a href="#{i}">{short(t)}</a></li>\n' for i, t in missing)

# ---- 2) new chapter
cid = "sft-recall-rehearsal-matched-compute"
title = ("67. SFT quietly erased v3&rsquo;s recall, a rehearsal fix brings it back, five eval bugs fall out of the "
         "investigation &mdash; and a fair transformer comparison says we win on samples, not on compute")
chapter = f'''
<h2 id="{cid}">{title}</h2>
<p class="date">2026-09-26 (overnight + day)</p>

<div class="setback"><p><b>SFT wiped out most of v3&rsquo;s recall.</b> Single-pass MQAR among-candidates on the 10B base is 100/100/100/72/38% at N=4/8/16/32/64; the finished <code>sft_v3_363m</code> (60M tok) scored 62/69/22/0/0. We paused the pipeline before ORPO. A 6M-token A/B from the base reproduced the drop and located it: the v3 mix (tools+coldstart+alpaca/2) gave 100/100/97/34/3 and <b>alpaca-only gave 100/100/97/34/6</b> &mdash; so it is SFT itself eroding high-N recall, not the tool data. (The v1 &ldquo;SFT keeps recall&rdquo; belief in <a href="#posttrain-orpo-probe-tokenizer">&sect;60</a> was read through split-replay, see below.)</p></div>

<div class="milestone"><p><b>MQAR rehearsal during SFT restores it, cheaply at low weight.</b> We added <code>mqar_aug</code> to <code>train_sft</code> (the pretraining MQAR aux, default off; smoke-tested) and swept its weight at 6M tokens with the fixed tool eval at n=200. Recall N32/N64: none 34/3, <b>w0.1 62/25</b>, w0.2 53/28, w0.3 56/22, w0.5 59/25, high-N-only (N16&ndash;32, w0.2) 66/28 &mdash; flat from 0.1 up. Tool skill (mode/tool/args/refuse) falls with weight: none .915/.893/.697/.923, <b>w0.1 .885/.861/.680/.897</b>, w0.2 .860/.844/.631/.846, w0.5 .865/.803/.574/.910. w0.1 keeps the whole recall gain for ~2&ndash;3 points. w0.05 and a second w0.2 seed were lost to the disk outage below.</p></div>

<p>The investigation shook out five bugs, all fixed and tested. (1) <code>recall_eval --replay split</code> is <b>invalid on any stack with attention</b>: pass 2 feeds only the query token and carries spiking state but no KV, so the v3 base &ldquo;scored&rdquo; 50/41/28/12/0 split vs 100/100/100/72/38 single-pass; split stays valid only for all-spiking 30M screens. (2) <code>tool_eval</code>/<code>head_eval</code> sliced the first N of a kind-grouped file, so every eval was 100% gold-ANSWER and <code>tool_acc</code> was 0/0 &mdash; tool calling had never been measured. (3) <code>train_sft</code> and <code>train_dpo</code> trained on the hash test split (2,454 tool items, 2,196 preference pairs). (4) <code>grammar.TokenTable.mask</code> cached on <code>id(grammar)</code>; CPython reuses freed ids, so one tool set&rsquo;s masks served another (FAIL 31/40 &rarr; PASS). (5) At 1B the checkpoint save died allocating pinned host memory; <code>_to_cpu</code> now falls back to a pageable copy.</p>

<p>v4 got its first hard throughput number. The 1B A&prime; shape (<a href="#v3-lands-review-bugs-launch-plan">&sect;66</a>) on the 5090 at 72 seqs/step: B4 2.2k, <b>B8 4.0k tok/s</b> (28.2GB), B12 spills (12.7GB allocated, 33.7 reserved) &mdash; overhead-bound. At 4.0k, 16B tokens is ~46 days, so levers must deliver &ge;1.55&times;. <code>expandable_segments</code> would fit B12 but fails under WSL (&ldquo;device not ready&rdquo;). The 30M depth screen closed NULL (L4 55.4 vs L2 55.8 val_ppl, same recall, 1.3&ndash;1.8&times; slower) &rarr; stay wide.</p>

<div class="setback"><p><b>A fair transformer comparison: we win on samples, not on compute.</b> A 30M, 2-layer transformer with the residual fix, same data/tokens/MQAR aux, lost at matched tokens (val_ppl 87.8 on the spiking-tuned NorMuon recipe, <b>68.8 on AdamW</b>, vs spiking 55.8) and sat at chance on recall &mdash; until a 600M-token run showed a clean induction-head phase transition at ~200M tokens (loss 8.0&rarr;4.98). The spiking model learns recall inside 100M. But the transformer is ~6&ndash;8&times; cheaper per token: in <b>1.36h</b> (less than spiking&rsquo;s 1.5h for 100M) it reached <b>val_ppl 39.7 and recall 100/97</b> vs spiking&rsquo;s 55.8 and 59/38. At 30M and with unoptimized spiking kernels, the honest pitch is sample efficiency (and energy, unmeasured) &mdash; not quality per GPU-hour. The real test is a v3-matched 426M transformer (17L, 421.5M, AdamW 3e-4); it probed at ~10.8k tok/s on the 5080, i.e. ~10.7 days for 10B tokens.</p></div>

<p>Process lessons: two 426M SFT runs on the 5090 spill past 32GB (run arms sequentially), and sharing the 5080 between a screen and an eval OOM&rsquo;d the screen. Then at 13:38 drive G:, which holds WSL&rsquo;s <code>ext4.vhdx</code>, hit 0 free &mdash; every WSL process, down to <code>sleep</code>, segfaulted until Glenn freed space. See <code>neuro_v3_sft_recall_loss_2026_09_26</code> + <code>reference_wsl_g_drive_full</code> + HANDOFF_2026_09_26.</p>

<hr>
<p class="date">Major version: 1.52 (v3 SFT erased recall [base 100/100/100/72/38 -&gt; 62/69/22/0/0]; alpaca-only loses it too -&gt; SFT itself, not the data mix. mqar_aug rehearsal in train_sft fixes it: recall flat w0.1-0.5, tool cost smallest at w0.1 [62/25; .885/.861/.680/.897 vs none .915/.893/.697/.923]. Bugs fixed: split-replay invalid on attention stacks; tool_eval slice all-ANSWER; SFT+ORPO test leakage; grammar mask cache id-reuse; 1B pinned-ckpt OOM. v4 1B A' B8 4.0k tok/s -&gt; 16B ~46d; depth NULL. 30M transformer [residual fix]: spiking wins matched tokens [55.8 vs 68.8 AdamW], transformer wins matched compute [39.7, recall 100/97 in 1.36h] with an induction transition at ~200M tok; 426M v3-matched transformer probed 10.8k tok/s on 5080 [~10.7d/10B]. G: disk-full outage killed w0.05 + w0.2-seed1. See neuro_v3_sft_recall_loss_2026_09_26 + reference_wsl_g_drive_full + HANDOFF_2026_09_26.)</p>
'''
anchor = "\n</body></html>"
assert s.count(anchor) == 1
# the newest chapter + its footer sit last before </body>; our chapter goes after them (chronological)
s = s.replace(anchor, chapter + anchor)
toc_line = f'  <li><a href="#{cid}">67. SFT erased recall, rehearsal fixes it; matched-compute transformer comparison</a></li>\n'
s = s[:toc_end] + add + toc_line + s[toc_end:] if False else s
toc_end = s.index("</ol>", s.index('<div class="toc">'))
s = s[:toc_end] + add + toc_line + s[toc_end:]
s, n = re.subn(r"Live document &mdash; last updated \d{4}-\d{2}-\d{2}|Live document — last updated \d{4}-\d{2}-\d{2}",
               lambda m: m.group(0)[:-10] + "2026-09-26", s)
assert n == 1
open(p, "w", encoding="utf-8").write(s)
print("TOC backfilled:", [i for i, _ in missing]); print("[OK] ch67 added")
