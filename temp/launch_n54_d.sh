#!/bin/bash
# launch_n54_d.sh -- 2026-10-10 (Neuro 54). One-shot: configs for the xchunk-start test, append r26s to the 5090 list,
# start the 5080 chain (7L pair) until Glenn's 17:30.
cd /home/glenn/projects/neuro; PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_p26_configs.py | tail -n 3
bash -n temp/queue_5080_n54.sh < /dev/null || exit 1
for r in screen_p26za_7L_RC3_x50 screen_p26zb_7L_RC3_x20 r26s_13L_RC3_x20_150m; do [ -f configs/$r.yaml ] || { echo "missing $r"; exit 1; }; done
diff configs/screen_p26za_7L_RC3_x50.yaml configs/screen_p26zb_7L_RC3_x20.yaml
grep -qx r26s_13L_RC3_x20_150m temp/queue_5090_n54.list || echo r26s_13L_RC3_x20_150m >> temp/queue_5090_n54.list
printf 'screen_p26za_7L_RC3_x50\nscreen_p26zb_7L_RC3_x20\n' > temp/queue_5080_n54.list; : > temp/queue_5080_n54.done
pgrep -f "^bash temp/queue_5080_n54.sh" > /dev/null || UNTIL="2026-10-10 17:30" setsid -f bash temp/queue_5080_n54.sh < /dev/null > /dev/null 2>&1
sleep 20
cat temp/queue_5090_n54.list
pgrep -af "^bash temp/(queue_50|alert_watch|watchdog_tps|kill_at_step)" | cut -c1-80
tail -n 2 logs/gap_queue.log | cut -c1-120
