#!/bin/bash
# swap_after_smoke.sh -- 2026-10-10 (Neuro 55): replace the waiting queue_5090_after_smoke.sh (r26v straight after
# r26u) with queue_5090_smoke1b_rerun.sh (smoke rerun, then r26v). Only the idle waiter is stopped; no run is touched.
cd /home/glenn/projects/neuro
p=$(pgrep -f "^bash temp/queue_5090_after_smoke.sh")
[ -n "$p" ] && kill -TERM $p && echo "stopped after_smoke waiter $p"
sleep 1
pgrep -f "^bash temp/queue_5090_after_smoke.sh" > /dev/null && { echo "after_smoke still alive, NOT launching"; exit 1; }
pgrep -f "^bash temp/queue_5090_smoke1b_rerun.sh" > /dev/null && { echo "rerun queue already running"; exit 0; }
setsid -f bash temp/queue_5090_smoke1b_rerun.sh < /dev/null > /dev/null 2>&1
sleep 1
pgrep -af "^bash temp/queue_5090"
