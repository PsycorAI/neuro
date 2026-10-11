#!/bin/bash
# restart_alert_watch.sh -- 2026-10-11 (Neuro 55): restart the supervisor after an edit (graceful: STOP file, then relaunch).
cd /home/glenn/projects/neuro
bash -n temp/alert_watch.sh || exit 1
touch temp/alert_watch.STOP
for i in $(seq 1 70); do pgrep -f "^bash temp/alert_watch.sh" > /dev/null || break; sleep 2; done
p=$(pgrep -f "^bash temp/alert_watch.sh"); [ -n "$p" ] && kill -TERM $p && sleep 1
setsid -f bash temp/alert_watch.sh < /dev/null > /dev/null 2>&1
sleep 2; pgrep -af "^bash temp/alert_watch.sh"; tail -n 2 logs/alerts.log | cut -c1-120
