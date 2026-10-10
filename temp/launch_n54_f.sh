#!/bin/bash
# launch_n54_f.sh -- 2026-10-10 (Neuro 54). One-shot: seed-1 repeats of the 7L pair onto the 5080 list; stop the p26za
# cross-chunk haystack eval that spills the 16 GB card (its wrapper then finishes nl_fact / xcopy and the chain goes on).
cd /home/glenn/projects/neuro; PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_p26_configs.py | tail -n 2
for r in screen_p26zc_7L_RC3_x50_s1 screen_p26zd_7L_RC3_x20_s1; do
  [ -f configs/$r.yaml ] || { echo "missing $r"; exit 1; }
  grep -qx $r temp/queue_5080_n54.list || echo $r >> temp/queue_5080_n54.list
done
pkill -TERM -f "temp/xchunk_haystack_eval.py screen_p26za_7L_RC3_x50="
sleep 30
cat temp/queue_5080_n54.list
pgrep -af "^bash temp/queue_5080_n54|eval.py screen_p26za" | cut -c1-110
grep '^step' logs/r26r_13L_RC3_seed1_150m.log | tail -n 1 | cut -c1-80
