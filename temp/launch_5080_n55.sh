#!/bin/bash
# launch_5080_n55.sh -- 2026-10-10 (Neuro 55): 5080 chain for the night + 10-11 (Glenn: free all night and day).
cd /home/glenn/projects/neuro
pgrep -f "^bash temp/queue_5080" > /dev/null && { echo "a 5080 queue is already running"; exit 1; }
UNTIL="2026-10-11 23:30" setsid -f bash temp/queue_5080_n54.sh < /dev/null > /dev/null 2>&1
sleep 20
pgrep -af "^bash temp/queue_5080|^bash temp/run_5080"; tail -n 2 logs/gap_queue.log | cut -c1-160
