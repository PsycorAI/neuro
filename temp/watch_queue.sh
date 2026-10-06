#!/bin/bash
# watch_queue.sh -- 2026-09-29 watchdog for temp/run_5080_sched.sh. Run in the background; it EXITS (= one
# notification) on the first event worth acting on, printing it:
#   exit 0  a run's evals finished (haystack line), a pause checkpoint, or "scheduler DONE"
#   exit 1  a run exited non-zero, or the scheduler is gone without DONE
#   exit 2  stall: during GPU hours (00:00-18:55) neither the queue log nor the newest screen log was written for 25 min
# Re-arm after each event:  bash temp/watch_queue.sh
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
start=$(wc -l < $LOG)
while true; do
  new=$(tail -n +$((start + 1)) $LOG)
  if echo "$new" | grep -qE 'exit=[1-9]'; then echo "RUN FAILED:"; echo "$new" | tail -8; exit 1; fi
  if echo "$new" | grep -qE 'haystack|paused at|scheduler DONE'; then echo "$new" | tail -8; exit 0; fi
  if ! pgrep -f run_5080_sched.sh >/dev/null; then echo "SCHEDULER NOT RUNNING (no DONE line):"; tail -5 $LOG; exit 1; fi
  # no stall check while the scheduler sleeps through Glenn's 19:00-24:00 window (false alarm at 00:00 on 09-30)
  if [ $((10#$(date +%H%M))) -lt 1855 ] && ! tail -1 $LOG | grep -q 'sleeping'; then
    newest=$(ls -t logs/screen_*.log | head -1)
    last=$(( $(stat -c %Y $newest) > $(stat -c %Y $LOG) ? $(stat -c %Y $newest) : $(stat -c %Y $LOG) ))
    age=$(( $(date +%s) - last ))
    if [ $age -gt 1500 ]; then echo "STALL? no log writes for $((age / 60)) min; newest $newest:"; tail -3 $newest; exit 2; fi
  fi
  sleep 60
done
