#!/bin/bash
# restart_convsmoke.sh -- 2026-10-11 (Neuro 55): relaunch the conversion-smoke driver after a script edit. Only while it
# is still in its wait loop (no conversion smoke trainer running).
cd /home/glenn/projects/neuro
pgrep -f "src/train_gpu.py --config configs/v4_smoke1b_conv_(t1024|noaux)" > /dev/null && { echo "its trainer is running, NOT restarting"; exit 1; }
p=$(pgrep -f "^bash temp/queue_5090_convsmoke.sh"); [ -n "$p" ] && kill -TERM $p && echo "stopped driver $p"
sleep 1
pgrep -f "^bash temp/queue_5090_convsmoke.sh" > /dev/null && { echo "driver still alive"; exit 1; }
bash -n temp/queue_5090_convsmoke.sh || exit 1
setsid -f bash temp/queue_5090_convsmoke.sh < /dev/null > /dev/null 2>&1
sleep 1; pgrep -af "^bash temp/queue_5090|^bash cloud/" | cut -c1-110
