#!/bin/bash
cd /home/glenn/projects/neuro
sed -i "s/$//" temp/queue_5080_r26i.sh temp/queue_5090_chain3.sh
bash -n temp/queue_5080_r26i.sh && bash -n temp/queue_5090_chain3.sh || exit 1
for p in $(pgrep -f "^bash temp/queue_5090_chain2.sh$"); do echo "TERM chain2 $p"; kill -TERM $p; done
sleep 1
setsid -f bash temp/queue_5090_chain3.sh > /dev/null 2>&1
setsid -f bash temp/queue_5080_r26i.sh > /dev/null 2>&1
echo "[n53 $(date +%m-%d\ %H:%M:%S)] RC2 = RC + HOLA (r26h win). 5080: r26i (RC2 + random chunks, expandable_segments). 5090 chain3 after r26f: r26j (RC2 conversion) -> r26g -> r26e" >> logs/gap_queue.log
sleep 2; ps -eo pid,args | grep -E "chain|queue_50|run_gap|sched|train_gpu" | grep -v grep | grep -v launch_ | cut -c1-120
