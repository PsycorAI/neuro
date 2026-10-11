#!/bin/bash
# cloud_watch.sh <ssh-host> <ssh-port> -- 2026-10-11 (Neuro 55). HOME-side monitor of the H200 run: until now nothing
# told anyone when the cloud run stopped (tripwire exit 2 = run_v4.sh does not retry; 5 crashes; seed failure; outbid).
# Every EVERY seconds it reads cloud/status.sh over ssh and raises alerts through temp/alert.sh (logs/alerts.log +
# Windows toast; a session's temp/wait_alert.sh wakes on them). Each condition alerts once (logs/cloud_watch.seen).
#   ALERT  instance unreachable 3 checks in a row | run_v4.sh not running before ALL DONE | tripwire tripped
#          | log silent >= STALL_MIN min while the runner is up | disk free < DISK_MIN GB
#   WARN   median tok/s < MIN_TPS (0 = off; set it to ~60% of the benched speed)
#   DONE   dense phase finished | all finished (then this script exits)
#   INFO   a status line every INFO_H hours
# Launch at home: setsid -f bash cloud/cloud_watch.sh <host> <port> < /dev/null > /dev/null 2>&1
# Test without an instance: CW_CMD="..." CW_ALERT=echo EVERY=1 bash cloud/cloud_watch.sh x x
H=${1:?ssh host}; P=${2:?ssh port}
cd /home/glenn/projects/neuro
EVERY=${EVERY:-600}; STALL_MIN=${STALL_MIN:-30}; MIN_TPS=${MIN_TPS:-0}; DISK_MIN=${DISK_MIN:-30}; INFO_H=${INFO_H:-6}
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o BatchMode=yes"
CMD=${CW_CMD:-$SSH root@$H bash /workspace/neuro/cloud/status.sh}
SEEN=${CW_SEEN:-logs/cloud_watch.seen}; touch $SEEN
A(){ ${CW_ALERT:-bash temp/alert.sh} "$@"; }
get(){ echo "$st" | sed -n "s/^$1=//p" | head -1; }
once(){ k=$1; shift; grep -qxF "$k" $SEEN && return 0; echo "$k" >> $SEEN; A "$@"; }
fails=0; next_info=0
while :; do
  st=$($CMD 2>/dev/null)
  if ! echo "$st" | grep -q '^runner='; then
    fails=$((fails + 1))
    [ $fails = 3 ] && A ALERT "cloud: instance unreachable for $((3 * EVERY / 60)) min (outbid, host down or ssh changed?)"
    sleep $EVERY; continue
  fi
  [ $fails -ge 3 ] && A INFO "cloud: instance reachable again"
  fails=0
  ph=$(get phase); sl=$(get step_line | cut -c1-60); now="$ph $sl | val $(get val)| $(get tps_med) tok/s"
  if [ "$(get all_done)" = 1 ]; then once "all_done" DONE "cloud: ALL DONE ($now)"; exit 0; fi
  [ "$(get dense_done)" = 1 ] && once "dense_done" DONE "cloud: dense phase DONE -> seeding + conversion ($(get runner_last | cut -c1-80))"
  [ "$(get trip)" -gt 0 ] 2>/dev/null && once "trip|$ph" ALERT "cloud: val_ppl TRIPWIRE tripped in $ph -- run stopped, needs a decision ($now)"
  if [ "$(get runner)" = 0 ]; then
    once "stopped|$(get runner_last)" ALERT "cloud: run_v4.sh is NOT running (last: $(get runner_last | cut -c1-90)) | $now"
  elif [ "$(get log_age_min)" -ge $STALL_MIN ] 2>/dev/null; then
    once "stall|$ph|$sl" ALERT "cloud: $ph log silent $(get log_age_min) min (trainer $(get trainer), gpu $(get gpu)) | $now"
  fi
  [ "$(get disk_free_gb)" -lt $DISK_MIN ] 2>/dev/null && once "disk|$(date +%d%H)" ALERT "cloud: only $(get disk_free_gb) GB free on the instance"
  t=$(get tps_med)
  [ $MIN_TPS -gt 0 ] && [ -n "$t" ] && [ "$t" -lt $MIN_TPS ] 2>/dev/null && once "slow|$(date +%d%H)" WARN "cloud: $t tok/s < $MIN_TPS ($now)"
  if [ $(date +%s) -ge $next_info ]; then
    A INFO "cloud: $now | of $(get total_steps) steps | disk $(get disk_free_gb) GB | gpu $(get gpu)"
    next_info=$(( $(date +%s) + INFO_H * 3600 ))
  fi
  sleep $EVERY
done
