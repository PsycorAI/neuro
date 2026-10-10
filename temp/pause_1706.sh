#!/bin/bash
cd /home/glenn/projects/neuro
for pat in "^bash temp/queue_5090_chain.sh$" "^bash temp/queue_5080_chain.sh$"; do for p in $(pgrep -f "$pat"); do echo "TERM $p: $(tr "\0" " " < /proc/$p/cmdline)"; kill -TERM $p; done; done
sleep 2
echo "[n53 $(date +%m-%d\ %H:%M:%S)] PAUSE (Glenn: disk space): chains stopped; r26d (5090) and p26y (5080) finish + save + evals, nothing new starts" >> logs/gap_queue.log
ps -eo pid,args | grep -E "queue|sched|train_gpu|chain" | grep -v grep | grep -v pause_ | cut -c1-150
