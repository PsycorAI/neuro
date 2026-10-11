#!/bin/bash
# launch_recover.sh -- 2026-10-10 (Neuro 55): after the power loss, restart the supervisor, the tok/s watchdog and the
# 5090 recovery chain. Starts nothing that is already running.
cd /home/glenn/projects/neuro
pgrep -f "train_gpu.py" > /dev/null && { echo "a trainer is already running, NOT launching"; exit 1; }
pgrep -f "^bash temp/queue_5090" > /dev/null && { echo "a 5090 queue is already running, NOT launching"; exit 1; }
pgrep -f "^bash temp/alert_watch.sh" > /dev/null || setsid -f bash temp/alert_watch.sh < /dev/null > /dev/null 2>&1
setsid -f bash temp/queue_5090_recover.sh < /dev/null > /dev/null 2>&1
sleep 3
pgrep -f "^bash temp/watchdog_tps.sh" > /dev/null || setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
sleep 2
pgrep -af "^bash temp/"
