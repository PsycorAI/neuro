#!/bin/bash
# launch_n54_b.sh -- 2026-10-09 (Neuro 54). One-shot: checks, regenerate configs, queue r26q (resume control) behind
# r26p, arm its planned stop, start the KD slice build.
cd /home/glenn/projects/neuro; PY=/home/glenn/projects/bdh/venv/bin/python
$PY temp/chk_valppl_only.py || exit 1
$PY temp/mk_p26_configs.py | tail -n 1
$PY temp/mk_v4_cloud.py | head -n 1
bash -n cloud/run_v4.sh || exit 1
[ -f configs/r26q_13L_RC3_resume10k_150m.yaml ] || { echo "no r26q config"; exit 1; }
grep -qx r26q_13L_RC3_resume10k_150m temp/queue_5090_n54.list || echo r26q_13L_RC3_resume10k_150m >> temp/queue_5090_n54.list
setsid -f bash temp/kill_at_step.sh r26q_13L_RC3_resume10k_150m 10000 < /dev/null > /dev/null 2>&1
setsid -f bash temp/run_kd_slice.sh < /dev/null > /dev/null 2>&1
sleep 3
cat temp/queue_5090_n54.list
pgrep -af "^bash temp/(queue_5090|alert_watch|watchdog_tps|kill_at_step|run_kd_slice)" | cut -c1-90
grep -E '^step' logs/r26p_13L_RC3_headwd_150m.log | tail -n 1 | cut -c1-90
grep -c watchdog_valppl configs/v4_cloud.yaml configs/v4_cloud_conv.yaml
tail -n 4 logs/alerts.log
