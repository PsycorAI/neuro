#!/bin/bash
# stop_recover.sh -- 2026-10-10 (Neuro 55): the tok/s watchdog killed the r26u resume at 21:29 (stale log mtime read as
# "silent 151 min"), so queue_5090_recover.sh went on to evaluate the step-15000 ckpt. Stop that chain (queue first,
# then its eval children); no trainer is running.
pgrep -f "train_gpu.py" > /dev/null && { echo "a trainer is running, NOT stopping anything"; exit 1; }
for pat in "^bash temp/queue_5090_recover.sh" "^bash temp/run_r25_evals.sh r26u_" "temp/[a-z0-9_]*eval[a-z0-9_]*.py r26u_"; do
  p=$(pgrep -f "$pat"); [ -n "$p" ] && kill -TERM $p && echo "stopped $pat: $p"
done
sleep 3
ps -eo pid,args | grep -E "temp/|train_" | grep -v grep | cut -c1-120
