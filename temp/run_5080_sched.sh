#!/bin/bash
# run_5080_sched.sh -- 2026-09-29. Screen queue that respects Glenn's GPU time (he uses the 5080 19:00-24:00 most nights).
# At PAUSE (18:55) the running job gets SIGINT: train_gpu finishes the step, saves a final ckpt, exits 130.
# The queue sleeps to 00:00, then relaunches the same run (train_gpu auto-resumes from the latest ckpt).
# No fresh start or eval after NOSTART (18:30). A finished run gets a DONE marker and is skipped on a re-launch.
# Waits (blocking flock) for any other queue on the same GPU (e.g. run_gap_queue.sh) to finish first.
# 2026-09-30 (Glenn: he sometimes uses the 5080 past midnight): before EVERY start/resume/eval the GPU must be FREE =
# utilization.gpu <= FREE_UTIL % for FREE_MIN consecutive minutes (checked once a minute; logs every 15 min while busy).
# Usage: GPU=1 bash temp/run_5080_sched.sh <run> [<run> ...]
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_queue.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU} PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_queue.log
PAUSE=${PAUSE:-1855}; NOSTART=${NOSTART:-1830}
FREE_UTIL=${FREE_UTIL:-5}; FREE_MIN=${FREE_MIN:-5}
say(){ echo "[s$GPU $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
hhmm(){ echo $((10#$(date +%H%M))); }
to_midnight(){ say "$1 -- GPU is Glenn's until 00:00, sleeping"; sleep $(( $(date -d 'tomorrow 00:00' +%s) - $(date +%s) + 30 )); }
wait_free(){   # $1 = what we are about to do
  local n=0 k=0 u
  while [ $n -lt $FREE_MIN ]; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
    if [ -n "$u" ] && [ "$u" -le "$FREE_UTIL" ]; then n=$((n + 1))
    else n=0; [ $((k % 15)) = 0 ] && say "$1 -- GPU $GPU in use (util ${u:-?}%), waiting for it to be free"; k=$((k + 1)); fi
    [ $n -lt $FREE_MIN ] && sleep 60
  done
}
exec 9>logs/gap_queue_gpu$GPU.lock; say "scheduler waiting for GPU $GPU lock"; flock 9
for run in "$@"; do
  ck=models/checkpoints/$run
  if [ -f $ck/DONE ]; then say "[$run] already DONE, skip"; continue; fi
  if ls $ck/step_*.pt >/dev/null 2>&1; then mode=RESUME; else mode=START; rm -rf $ck; : > logs/$run.log; fi
  while :; do
    [ $(hhmm) -ge $NOSTART ] && to_midnight "[$run] before $mode"
    wait_free "[$run] before $mode"
    [ $(hhmm) -ge $NOSTART ] && continue          # the wait ran past NOSTART -> back to the midnight sleep
    say "[$run] $mode"
    "$PY" src/train_gpu.py --config configs/$run.yaml --device cuda >> logs/$run.log 2>&1 &
    pid=$!; paused=0
    while kill -0 $pid 2>/dev/null; do
      if [ $paused = 0 ] && [ $(hhmm) -ge $PAUSE ]; then say "[$run] PAUSE (SIGINT -> final ckpt)"; kill -INT $pid; paused=1; fi
      sleep 20
    done
    wait $pid; rc=$?
    if [ $rc = 130 ]; then say "[$run] paused at $(ls $ck/step_*.pt | tail -1)"; mode=RESUME; continue; fi
    break
  done
  say "[$run] exit=$rc :: $(grep 'eval\] val_ppl' logs/$run.log | awk '{print $3}' | tr '\n' ' ')"
  [ $rc = 0 ] || continue
  touch $ck/DONE
  while :; do
    [ $(hhmm) -ge $NOSTART ] && to_midnight "[$run] before evals"
    wait_free "[$run] before evals"
    [ $(hhmm) -ge $NOSTART ] || break
  done
  last=$(ls $ck/step_*.pt 2>/dev/null | tail -1)
  "$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 | sed "s/^/  recall /" >> "$LOG"
  [ -n "$last" ] && "$PY" temp/haystack_recall_eval.py $run=$last 2>&1 | grep -E "^$run" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> "$LOG"
done
say "scheduler DONE"
