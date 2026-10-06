#!/bin/bash
# relaunch_5080_r25.sh -- 2026-10-05: restart the (waiting) 5080 r25 queue with the edited script + start the watchdog.
cd /home/glenn/projects/neuro
for p in $(pgrep -f "bash temp/queue_5080_r25.sh"); do kill $p; done
for p in $(pgrep -f "flock logs/gap_queue_gpu1.lock true"); do kill $p; done
sleep 1
setsid -f bash temp/queue_5080_r25.sh > /dev/null 2>&1
pgrep -f "bash temp/watchdog_tps.sh" > /dev/null || setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
sleep 2
ps -eo pid,args | grep -E "bash temp/(queue_50|watchdog|run_)|flock logs" | grep -v grep
tail -3 logs/gap_queue.log
