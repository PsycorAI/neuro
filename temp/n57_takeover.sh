#!/bin/bash
# Neuro 57 takeover check: read-only.
cd /home/glenn/projects/neuro || exit 1
echo "== date"; date
echo "== waiters"; pgrep -af '^bash temp/wait_alert.sh' || echo none
echo "== home scripts"
pgrep -af '^bash cloud/cloud_watch.sh' || echo "cloud_watch MISSING"
pgrep -af '^bash cloud/pull_ckpts.sh' || echo "pull_ckpts MISSING"
pgrep -af '^bash temp/alert_watch.sh' || echo "alert_watch MISSING"
pgrep -af '^bash temp/watchdog_tps.sh' || echo "watchdog_tps MISSING"
pgrep -af '^bash temp/queue_5080_sftreh.sh' || echo "sftreh driver gone (expected after its DONE)"
echo "== alerts tail"; tail -n 15 logs/alerts.log
echo "== pull_ckpts tail"; tail -n 6 logs/pull_ckpts.log
echo "== sft evals tail"; tail -n 45 logs/sft_v4_reh13_evals.log
echo "== gpus"; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
echo "== disk"; df -h /home | tail -n 1
true
