#!/bin/bash
# launch_n54_e.sh -- 2026-10-10 (Neuro 54). One-shot: r26t (resume control with the fix) goes BEFORE r26s in the 5090
# list; arm its planned stop.
cd /home/glenn/projects/neuro; PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/mk_p26_configs.py | grep -c wrote
[ -f configs/r26t_13L_RC3_resume10k_fix_150m.yaml ] || { echo "missing r26t"; exit 1; }
git status --short configs | grep -v '^??'
printf 'r26p_13L_RC3_headwd_150m\nr26q_13L_RC3_resume10k_150m\nr26r_13L_RC3_seed1_150m\nr26t_13L_RC3_resume10k_fix_150m\nr26s_13L_RC3_x20_150m\n' > temp/queue_5090_n54.list
pgrep -f "^bash temp/kill_at_step.sh r26t_" > /dev/null || setsid -f bash temp/kill_at_step.sh r26t_13L_RC3_resume10k_fix_150m 10000 < /dev/null > /dev/null 2>&1
sleep 2
cat temp/queue_5090_n54.done; echo ---; pgrep -af "^bash temp/(queue_50|alert_watch|watchdog_tps|kill_at_step)" | cut -c1-90
grep '^step' logs/r26r_13L_RC3_seed1_150m.log | tail -n 1 | cut -c1-80
grep '^step' logs/screen_p26za_7L_RC3_x50.log | tail -n 1 | cut -c1-80
