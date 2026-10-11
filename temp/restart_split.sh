#!/bin/bash
# restart_split.sh -- 2026-10-11 (Neuro 55): relaunch temp/queue_5090_split.sh after an edit. Only while it is still
# waiting for the n54 chain (none of its GPU arms running).
cd /home/glenn/projects/neuro
pgrep -f "temp/profile_train.py .*--tag (split_|prof_|t_)" > /dev/null && { echo "its GPU work is running, NOT restarting"; exit 1; }
pgrep -f "^bash temp/queue_5090_n54.sh" > /dev/null || { echo "the n54 chain is not running: the driver may be past its wait, NOT restarting"; exit 1; }
p=$(pgrep -f "^bash temp/queue_5090_split.sh"); [ -n "$p" ] && kill -TERM $p && echo "stopped driver $p"
sleep 1
pgrep -f "^bash temp/queue_5090_split.sh" > /dev/null && { echo "driver still alive"; exit 1; }
bash -n temp/queue_5090_split.sh || exit 1
setsid -f bash temp/queue_5090_split.sh < /dev/null > /dev/null 2>&1
sleep 1; pgrep -af "^bash temp/queue_5090" | cut -c1-100; grep -c HOLD temp/queue_5090_n54.list
