#!/bin/bash
# alert_watch.sh -- 2026-10-09 (Neuro 54; Glenn: robust chain + intelligent watchdog + alerts, no manual monitoring).
# Independent supervisor (not part of any queue). Every 2 min:
#  1. new lines of logs/gap_queue.log: crash / kill / trip markers                       -> ALERT
#  2. each live trainer: nan loss -> ALERT; val_ppl rising over 3 evals -> WARN; flat (<0.5% better over 2 evals past
#     75% of the run, the r26k / r26m pattern) -> WARN
#  3. a trainer is alive but temp/watchdog_tps.sh is not                                  -> restart it + WARN
#  4. a queue is alive but nothing under logs/ changed for 45 min (hung chain)            -> ALERT
#  5. no queue and no trainer for 15 min                                                  -> INFO (GPUs idle), once
#  6. G: < 40 GB or / < 20 GB free                                                        -> ALERT (once per 6 h)
# Each condition alerts once per episode (keys in temp/alert_watch.seen). Heartbeat: logs/alert_watch.heartbeat.
# Stop: touch temp/alert_watch.STOP    Launch: setsid -f bash temp/alert_watch.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; SEEN=temp/alert_watch.seen; touch $SEEN; rm -f temp/alert_watch.STOP
A(){ bash temp/alert.sh "$@"; }
once(){ grep -qxF "$1" $SEEN && return 1; echo "$1" >> $SEEN; return 0; }      # true the first time a key is seen
n0=$(wc -l < $LOG); idle=0
A INFO "alert_watch started"
while [ ! -f temp/alert_watch.STOP ]; do
  date +%s > logs/alert_watch.heartbeat
  # 1. queue-log markers (new lines only; the chain's own [n54 lines report through alert.sh already)
  n1=$(wc -l < $LOG)
  if [ "$n1" -gt "$n0" ]; then
    sed -n "$((n0 + 1)),${n1}p" $LOG | grep -v '^\[n54' | grep -E 'exit=[1-9]|SIGTERM|TRIPPED|ENDED WITHOUT|FAILED|giving up|Traceback|log silent' \
      | cut -c1-170 | while read -r l; do A ALERT "queue log: $l"; done
    n0=$n1
  fi
  # 2. trainer health
  tr=0
  for pid in $(pgrep -f "[s]rc/train_gpu.py --config"); do
    tr=1
    r=$(tr '\0' ' ' < /proc/$pid/cmdline | sed -n 's/.*--config configs\/\([^ ]*\)\.yaml.*/\1/p'); lg=logs/$r.log
    [ -f "$lg" ] || continue
    st=$(grep '^step' $lg | tail -1 | awk '{print $2}')
    grep '^step' $lg | tail -3 | grep -qiE 'loss (nan|inf)' && once "nan $r" && A ALERT "$r: loss is nan/inf at step $st"
    v=$(grep 'eval\] val_ppl' $lg | awk '{print $3}' | tail -3 | tr '\n' ' ')
    tot=$(grep -o 'total_steps≈[0-9]*' $lg | tail -1 | grep -o '[0-9]*$')
    verdict=$(echo "$v ${st:-0} ${tot:-0}" | awk 'NF==5{ if ($2 > $1 && $3 > $2) print "RISING";
                 else if ($5 > 0 && $4 > 0.75 * $5 && $3 > 0.995 * $1) print "FLAT" }')
    [ -n "$verdict" ] && once "$verdict $r $v" && A WARN "$r val_ppl $verdict over the last 3 evals: $v(step $st of $tot)"
  done
  # 3. tok/s watchdog must be alive while anything trains
  if [ $tr = 1 ] && ! pgrep -f "^bash temp/watchdog_tps.sh" > /dev/null; then
    MIN_TPS=2500 setsid -f bash temp/watchdog_tps.sh < /dev/null > /dev/null 2>&1
    A WARN "watchdog_tps.sh was not running while a run trains -> restarted (MIN_TPS 2500)"
  fi
  # 4. / 5. hung chain, idle GPUs
  q=0; pgrep -f "[q]ueue_50[89]0_|[q]ueue_spk_|[r]un_gap_queue.sh|[r]un_5080_sched.sh" > /dev/null && q=1
  newest=$(ls -t logs/*.log | grep -v alerts.log | head -1); age=$(( ($(date +%s) - $(stat -c %Y "$newest")) / 60 ))
  if [ $q = 1 ] && [ $age -ge 45 ]; then once "hung $(stat -c %Y "$newest")" && A ALERT "a queue is alive but no log changed for $age min (hung chain?)"; fi
  if [ $q = 0 ] && [ $tr = 0 ]; then idle=$((idle + 1)); [ $idle = 8 ] && A INFO "no queue and no trainer for 15 min: GPUs idle"
  else idle=0; fi
  # 6. disk
  for m in /mnt/g:40 /:20; do
    free=$(df -BG --output=avail ${m%%:*} 2>/dev/null | tail -1 | tr -dc '0-9')
    [ -n "$free" ] && [ "$free" -lt ${m##*:} ] && once "disk ${m%%:*} $(( $(date +%s) / 21600 ))" && A ALERT "disk ${m%%:*} has only ${free} GB free"
  done
  sleep 120
done
A INFO "alert_watch stopped (STOP file)"
