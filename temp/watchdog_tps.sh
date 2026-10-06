#!/bin/bash
# watchdog_tps.sh -- 2026-10-05 (Glenn: robust chains + watchdogs, no manual babysitting).
# Every 5 min, for each live train_gpu.py run: if the median tok/s of its last 10 log lines (past step 1000) is below
# MIN_TPS -- a WDDM VRAM spill or a hang -- the run is killed with SIGTERM (not SIGINT: the 5080 scheduler treats
# exit 130 as a pause and would resume it). Its queue then moves to the next run. Also kills a run whose log has not
# changed for STALL_MIN minutes (hang). Logs to logs/gap_queue.log. Exits when no queue is left.
# Launch: setsid -f bash temp/watchdog_tps.sh > /dev/null 2>&1
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
MIN_TPS=${MIN_TPS:-8000}; STALL_MIN=${STALL_MIN:-25}
say(){ echo "[wd $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
say "watchdog started (min ${MIN_TPS} tok/s past step 1000, stall ${STALL_MIN} min)"
while pgrep -f "[q]ueue_50[89]0_|[r]un_gap_queue.sh|[r]un_5080_sched.sh" > /dev/null; do
  for pid in $(pgrep -f "[s]rc/train_gpu.py"); do
    cfg=$(tr '\0' ' ' < /proc/$pid/cmdline | sed -n 's/.*--config configs\/\([^ ]*\)\.yaml.*/\1/p')
    [ -z "$cfg" ] && continue
    lg=logs/$cfg.log; [ -f "$lg" ] || continue
    age=$(( ($(date +%s) - $(stat -c %Y "$lg")) / 60 ))
    if [ $age -ge $STALL_MIN ]; then say "[$cfg] log silent ${age} min -> SIGTERM $pid"; kill -TERM $pid; continue; fi
    # 10-05: 5 lines past step 300 (p25f spilled at 2.4k tok/s for 3.5 h and never reached 10 lines past step 1000)
    # only lines after the latest (re)start in the log (a resumed run appends to the old log)
    med=$(awk '/^resumed |^arch=/{n=0; delete a} /^step [0-9]+ \|/{a[++n]=$0} END{for(i=1;i<=n;i++) print a[i]}' "$lg" \
          | awk '{s=$2} s>300' | tail -5 | sed -n 's/.*| \([0-9,]*\) tok\/s.*/\1/p' \
          | tr -d ',' | sort -n | awk '{a[NR]=$1} END{if (NR>=5) print a[int((NR+1)/2)]}')
    if [ -n "$med" ] && [ "$med" -lt $MIN_TPS ]; then
      say "[$cfg] median ${med} tok/s < ${MIN_TPS} (spill/hang?) -> SIGTERM $pid"; kill -TERM $pid
    fi
  done
  sleep 300
done
say "watchdog exit (no queues left)"
