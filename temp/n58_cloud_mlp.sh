#!/bin/bash
# Neuro 58: read-only. Per-eval history of the cloud run: val, spike, the 12 attention-FFN activity rates, weight norms.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
$SSH root@$H 'cd /workspace/neuro; grep -E "\[eval\]|\[mlp\]|\[wnorm\]" logs/v4_cloud.log | sed -E "s/ thr=[0-9.]+//g; s/rate=//g; s/ablated_ppl [0-9.]+ //; s/\| target.*//" | paste - - - | cut -c1-250' 2>/dev/null
echo "== 13L reference r26p"
grep -a -E "\[eval\]|\[mlp\]" logs/r26p_13L_RC3_headwd_150m.log | sed -E "s/ thr=[0-9.]+//g; s/rate=//g; s/ablated_ppl [0-9.]+ //; s/\| target.*//" | paste - - | cut -c1-200
