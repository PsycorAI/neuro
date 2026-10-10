#!/bin/bash
# wait_alert.sh -- 2026-10-09 (Neuro 54). Session waiter: block until a NEW line of logs/alerts.log (appended after this
# starts) has level ALERT / WARN / DONE, or $1 minutes pass (default 110 < the 2 h background limit), then print the new
# lines. Also reports a dead supervisor (stale heartbeat). Never self-matches (no pgrep). Run with run_in_background.
cd /home/glenn/projects/neuro
touch logs/alerts.log; n0=$(wc -l < logs/alerts.log); end=$(( $(date +%s) + ${1:-110} * 60 ))
until tail -n +$((n0 + 1)) logs/alerts.log | grep -qE '^\[(ALERT|WARN|DONE) ' || [ "$(date +%s)" -ge "$end" ]; do
  hb=$(cat logs/alert_watch.heartbeat 2>/dev/null || echo 0)
  [ $(( $(date +%s) - hb )) -gt 900 ] && { echo "SUPERVISOR DEAD: alert_watch heartbeat is stale"; break; }
  sleep 30
done
date; tail -n +$((n0 + 1)) logs/alerts.log | cut -c1-300
tail -n +$((n0 + 1)) logs/alerts.log | grep -qE '^\[(ALERT|WARN|DONE) ' || echo "no ALERT/WARN/DONE in the window"
exit 0
