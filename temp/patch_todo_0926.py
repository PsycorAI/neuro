# patch_todo_0926.py -- add the 2026-09-26 overnight post-training findings to docs/to-do.html
p = "docs/to-do.html"
s = open(p, encoding="utf-8").read()
anchor = "</ol>\n<h2>Research Backlog</h2>"
assert s.count(anchor) == 1
new = """</ol>
<h2>Post-training v3 &mdash; overnight 2026-09-26 findings</h2>
<ol>
  <li><b style="color:#c0392b">v3 SFT destroyed recall</b> (single-pass MQAR N4/8/16/32/64): base <code>flagship_v3_363m</code> 100/100/100/72/38% &rarr; <code>sft_v3_363m</code> 62/69/22/0/0%. v1 SFT (alpaca-only) kept 100/100/81/44/19. Same LR/optimizer; differences = data mix (tools+coldstart+alpaca/2), n_neurons, chunk 256. Pipeline STOPPED before ORPO (Glenn 00:35). A/B screen (alpaca-only vs mix, 6M tok, from the 10B base) RUNNING: <code>temp/run_sftab.sh</code> &rarr; <code>logs/sftab_screen.log</code>.</li>
  <li><b>recall_eval <code>--replay split</code> is INVALID for attention stacks</b> (v3 has 1 attn layer): pass 2 feeds only the query token and carries only spiking state, not KV &rarr; v3 base scored 50/41/28/12/0 split vs 100/100/100/72/38 single-pass. Use single-pass for v3/v4/transformers; split stays valid for the all-spiking 30M screens. <b>TO-FIX</b> in <code>run_v3_posttrain.sh</code> eval block (drop <code>--replay split</code>).</li>
  <li><b>tool_eval measured nothing</b>: tools.jsonl is grouped by kind, so the first-200 test slice was 100% gold-ANSWER &rarr; <code>tool_acc</code>=0/0. FIXED (seeded shuffle) in <code>tool_eval.py</code> + <code>head_eval.py</code>.</li>
  <li><b>Test-set leakage into training</b>: <code>train_sft</code> loaded tools.jsonl with no split (trained on the 2,454 held-out test items); ORPO loaded why/abstain.jsonl with no split (2,196 test pairs). FIXED: <code>split="train"</code> in train_sft; <code>drop_test=True</code> in train_dpo (test: <code>temp/test_dpo_drop_test.py</code> PASS).</li>
  <li><b>30M FIXED-transformer preview</b> (<code>screen_tfmrfix_s0_30m</code>, 2L d1024 all-attn + attn_single_residual, 58.75M vs spiking control 59.58M, same data/MQAR/seed) queued on the 5080 after depth: <code>logs/tfmrfix_screen.log</code>. Early read on the invalidated 34% claim (the real one = 363M Phase 0 #1).</li>
  <li><b>1B A&prime; tok/s+VRAM probe</b> built: <code>temp/run_1b_probe.sh</code> (B 12/8/4 at 72 seqs/step) &rarr; <code>logs/probe_1b.log</code>. Runs on the 5090 after the A/B.</li>
</ol>
<h2>Research Backlog</h2>"""
s = s.replace(anchor, new)
open(p, "w", encoding="utf-8").write(s)
print("[OK] to-do updated")
