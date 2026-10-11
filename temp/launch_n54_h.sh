#!/bin/bash
# launch_n54_h.sh -- 2026-10-10 (Neuro 54). One-shot: r26v config + list entry, chain it after the smoke queue.
cd /home/glenn/projects/neuro; PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_p26_configs.py | grep -c wrote
r=r26v_13L_RC3_gap20_x20_150m
[ -f configs/$r.yaml ] || { echo "missing $r"; exit 1; }
diff configs/r26n_13L_RC2_xlate_150m.yaml configs/$r.yaml | grep '^[<>]' | cut -c1-70
grep -qx $r temp/queue_5090_n54.list || echo $r >> temp/queue_5090_n54.list
bash -n temp/queue_5090_after_smoke.sh < /dev/null || exit 1
pgrep -f "^bash temp/queue_5090_after_smoke.sh" > /dev/null || setsid -f bash temp/queue_5090_after_smoke.sh < /dev/null > /dev/null 2>&1
sleep 2
pgrep -af "^bash temp/(queue_50|alert_watch|watchdog_tps|kill_at_step)" | cut -c1-80
tail -n 2 logs/run_v4.log | cut -c1-120
grep -E '^arch=|^step|Error|Traceback' logs/v4_smoke1b.log 2>/dev/null | tail -n 3 | cut -c1-170
