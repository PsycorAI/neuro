#!/bin/bash
# stop only the outer 5080 r25 queue loop (the running run_5080_sched child + its train keep going)
for p in $(pgrep -f "bash temp/queue_5080_r25.sh"); do kill $p; done
sleep 1; ps -eo pid,args | grep -E "bash temp/(queue_50|run_5080|watchdog)|train_gpu" | grep -v grep | cut -c1-120
