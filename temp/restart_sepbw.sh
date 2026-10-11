#!/bin/bash
# restart_sepbw.sh -- 2026-10-11 (Neuro 55): relaunch the separate-backward validation driver after a script edit.
# Only while it is still waiting (nothing of its own running on the GPU).
cd /home/glenn/projects/neuro
pgrep -f "temp/profile_train.py .*--tag sep_|configs/v4_smoke1b_conv_sepbw" > /dev/null && { echo "its GPU work is running, NOT restarting"; exit 1; }
p=$(pgrep -f "^bash temp/queue_5090_sepbw.sh"); [ -n "$p" ] && kill -TERM $p && echo "stopped driver $p"
sleep 1
pgrep -f "^bash temp/queue_5090_sepbw.sh" > /dev/null && { echo "driver still alive"; exit 1; }
bash -n temp/queue_5090_sepbw.sh || exit 1
setsid -f bash temp/queue_5090_sepbw.sh < /dev/null > /dev/null 2>&1
sleep 1; pgrep -af "^bash temp/queue_5090|^bash temp/kill_at_step" | cut -c1-100
