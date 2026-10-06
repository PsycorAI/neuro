p = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(p).read()
anchor = "<li><b>IDEA SWEEP 2026-09-29"
assert s.count(anchor) == 1
new = ('<li><b style="color:green">GENTLE GAP CURRICULUM WORKS (r7_gapw015, 2026-09-30, 1 seed, 5080)</b>: '
       'unsigned hybrid full stack + haystack MQAR gap 440 at mqar_weight 0.15 &rarr; val_ppl <b>71.10</b> '
       '(control full stack ~74.9 s1; r6 gap at full weight 92.55) = NO ppl cost; recall 100/100/97/100/94; '
       '<b>haystack SOLVED</b> N8 gap0/64/192/440 99/100/99/100, N32 96/100/99/100 (signed+canon: N32 gap440 18.4); '
       'memory half-life on text ~3700/2900 tok (was ~1); mem_off = chance, attn_off unchanged; firing 5.3% '
       '(mean level 1.14). <b>NEXT (TO-TEST, needs OK):</b> signed+canon + gap440 w0.15 (combine best ppl with haystack), '
       'then a seed-2 confirmation. Logs: <code>logs/gap_queue.log</code>, <code>logs/diag_haystack_pathways.log</code>, '
       '<code>logs/stream_eval_baseline.log</code>, <code>logs/diag_spike_levels.log</code>.</li>\n  ')
s = s.replace(anchor, new + anchor)
open(p, "w").write(s)
print("[OK] to-do updated")
