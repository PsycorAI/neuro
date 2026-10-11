#!/bin/bash
# restart_watchdog.sh -- 2026-10-11 (Neuro 55): restart temp/watchdog_tps.sh after an edit (same MIN_TPS as alert_watch uses).
cd /home/glenn/projects/neuro
bash -n temp/watchdog_tps.sh || exit 1
p=$(pgrep -f "^bash temp/watchdog_tps.sh"); [ -n "$p" ] && kill -TERM $p && echo "stopped watchdog $p"
sleep 1
MIN_TPS=${MIN_TPS:-2500} setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
sleep 1; pgrep -af "^bash temp/watchdog_tps.sh"; tail -n 2 logs/gap_queue.log | cut -c1-140
