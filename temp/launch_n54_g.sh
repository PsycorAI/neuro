#!/bin/bash
# launch_n54_g.sh -- 2026-10-10 (Neuro 54). One-shot: generate the 1B smoke + r26u configs, launch the queue that runs
# them after the n54 chain.
cd /home/glenn/projects/neuro; PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_v4_cloud.py 2>&1 | grep -E '^\[OK\]|Error'
$PY temp/mk_p26_configs.py | tail -n 1
bash -n temp/queue_5090_smoke1b.sh < /dev/null || exit 1
for f in v4_smoke1b v4_smoke1b_conv r26u_13L_RC3_conv_llamakd_s1_150m; do [ -f configs/$f.yaml ] || { echo "missing $f"; exit 1; }; done
diff configs/v4_cloud.yaml configs/v4_smoke1b.yaml | grep '^[<>]' | tr '\n' ' '; echo
diff configs/r26o_13L_RC3_conv_llamakd_150m.yaml configs/r26u_13L_RC3_conv_llamakd_s1_150m.yaml | grep '^[<>]' | cut -c1-90
git status --short configs | grep -v '^??'
pgrep -f "^bash temp/queue_5090_smoke1b.sh" > /dev/null || setsid -f bash temp/queue_5090_smoke1b.sh < /dev/null > /dev/null 2>&1
sleep 2
pgrep -af "^bash temp/(queue_50|alert_watch|watchdog_tps|kill_at_step)" | cut -c1-90
grep '^step' logs/r26t_13L_RC3_resume10k_fix_150m.log | tail -n 1 | cut -c1-80
