#!/bin/bash
# launch_n54_c.sh -- 2026-10-09 (Neuro 54). One-shot: val tail file, r26r config + append to the 5090 chain list, status.
cd /home/glenn/projects/neuro; PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_val_tail.py
$PY temp/mk_p26_configs.py | tail -n 1
bash -n cloud/pull_corpus.sh < /dev/null || exit 1
[ -f configs/r26r_13L_RC3_seed1_150m.yaml ] && { grep -qx r26r_13L_RC3_seed1_150m temp/queue_5090_n54.list || echo r26r_13L_RC3_seed1_150m >> temp/queue_5090_n54.list; }
cat temp/queue_5090_n54.list
pgrep -af "^bash temp/(queue_50|alert_watch|watchdog_tps|kill_at_step)" | cut -c1-80
grep '^step' logs/r26p_13L_RC3_headwd_150m.log | tail -n 1 | cut -c1-80
grep -E '^step|Error|Traceback' logs/v4_reh.log 2>/dev/null | tail -n 2 | cut -c1-110
tail -n 3 logs/run_v4.log 2>/dev/null
