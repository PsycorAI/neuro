#!/bin/bash
cd /home/glenn/projects/neuro
for p in $(pgrep -f "^bash temp/watchdog_tps.sh$"); do echo "TERM watchdog $p"; kill -TERM $p; done
echo "[n53 $(date +%m-%d\ %H:%M:%S)] watchdog stopped: r26d at 8.4k tok/s (30.8 GB reserved) is the last 5090 run; resume_after_cleanup.sh restarts it" >> logs/gap_queue.log
sed -n 1,20p temp/chk_5090.txt
