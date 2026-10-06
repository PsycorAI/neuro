"""todo_1006.py -- handoff 10-06: one consolidated status entry for every idea of Neuro 51 (above the 10-06 energy entry)."""
P = "/home/glenn/projects/neuro/docs/to-do.html"
s = open(P).read()
anchor = '  <li><b style="color:#2471a3">10-06 ENERGY PLAN (Glenn)'
add = ('  <li><b style="color:#c0392b">10-06 STATUS of every Neuro-51 idea (journey &sect;77; docs/memory_beyond_window_2026_10_05.md, '
       'docs/energy_plan_2026_10_06.md)</b>: '
       '<b>WIN</b>: chunked aligned attention (<code>attn_chunk</code>) + ONE spiking memory block at mid-depth, added (r25o); '
       'spike conversion of every linear input (<code>spike_code</code>, r25t L8 80M = no ppl cost, 5.0x). '
       '<b>BUILT/in r25w</b>: q/v spike-driven attention (6.2x), self-KD (<code>kd_self_ckpt</code>, Liger bug fixed), 1.5x thresholds. '
       '<b>TESTING</b>: factorised head <code>head_rank</code> 256/512 (r25y). '
       '<b>TO-TEST next</b>: way 2 = 2nd memory block at depth (14L, depths 4 + 9); T2048 (4 chunks) at 12L (p25q 7L worked); '
       'window-count cause of the +7.5% chunking cost (open); spike-count penalty instead of fixed thresholds (idea); drop the '
       'attention gate (6.7 uJ, idea). <b>OPEN #1</b>: real-text recall beyond the window (xcopy 30 vs 28 mem-off). '
       '<b>CLOSED-NEG</b>: SWAX r24a, stateful TBPTT r24b, late phases / late insertion (r25f-m), xcopy augmentation r25v (+9%, no gain), '
       'NL-fact aux (no gain), ternary spikes (+89%), spike-from-scratch p25t (dropped: conversion works). '
       '<b>PARKED</b>: multi-reader H0, Titans-MAC tokens, surprise-gated writes. '
       '<b>DEMO to-build</b>: chunked streaming (<code>scripts/demo/chunk_stream.py</code> built + verified) into server.py; '
       'key&ndash;value memory game. <b>v4</b>: write <code>configs/v4_cloud.yaml</code> (r25o layout at 1B + chunks + final '
       'conversion phase) + /audit-config, freeze ~Oct 8.</li>\n')
assert anchor in s and "STATUS of every Neuro-51" not in s
open(P, "w").write(s.replace(anchor, add + anchor, 1))
print("[OK]")
