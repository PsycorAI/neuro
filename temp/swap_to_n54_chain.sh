#!/bin/bash
# swap_to_n54_chain.sh -- 2026-10-09 (Neuro 54). One-shot launcher: supervisor + the robust chain. r26p already trains
# under queue_5090_r26p.sh (left alone); the chain waits for that queue, auto-resumes r26p if it did not finish, and
# reports the verdict.
cd /home/glenn/projects/neuro
for f in temp/alert.sh temp/alert_watch.sh temp/queue_5090_n54.sh temp/wait_alert.sh; do bash -n $f || exit 1; done
echo r26p_13L_RC3_headwd_150m > temp/queue_5090_n54.list; : > temp/queue_5090_n54.done
setsid -f bash temp/alert_watch.sh < /dev/null > /dev/null 2>&1
setsid -f bash temp/queue_5090_n54.sh < /dev/null > /dev/null 2>&1
sleep 3
pgrep -af "^bash temp/(queue_5090|alert_watch|watchdog_tps)" | cut -c1-80
bash temp/alert.sh DONE "test: alerts are live (r26p training on the 5090)"
tail -n 3 logs/alerts.log
grep -E 'head_weight_decay|^step' logs/r26p_13L_RC3_headwd_150m.log | tail -n 2 | cut -c1-120
