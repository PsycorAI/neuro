#!/bin/bash
# fix_0505.sh -- watchdog: watch while ANY temp/queue_50[89]0_* script or run queue is alive (generic pattern).
cd /home/glenn/projects/neuro
for p in $(pgrep -f "bash temp/watchdog_tps.sh"); do kill $p; done
sed -i 's/^while pgrep -f "[^"]*" > \/dev\/null; do$/while pgrep -f "[q]ueue_50[89]0_|[r]un_gap_queue.sh|[r]un_5080_sched.sh" > \/dev\/null; do/' temp/watchdog_tps.sh
grep -n '^while pgrep' temp/watchdog_tps.sh
bash -n temp/watchdog_tps.sh && setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
