#!/bin/bash
# run_gap_queue.sh -- 2026-09-28 generic ppl-gap screen queue (v2: adds haystack recall-under-interference eval).
# Usage: GPU=<0|1> DEADLINE=<HHMM start cutoff, 2400=none> bash temp/run_gap_queue.sh <run> [<run> ...]
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_queue.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU} PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_queue.log
exec 9>logs/gap_queue_gpu$GPU.lock; flock -n 9 || { echo "queue on GPU $GPU already running"; exit 1; }
say(){ echo "[q$GPU $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
for run in "$@"; do
  if [ "$(date +%H%M)" -gt "${DEADLINE:-2400}" ]; then say "[$run] SKIPPED (past start deadline $DEADLINE)"; continue; fi
  rm -rf models/checkpoints/$run
  say "[$run] START"
  "$PY" src/train_gpu.py --config configs/$run.yaml --device cuda > logs/$run.log 2>&1
  say "[$run] exit=$? :: $(grep 'eval\] val_ppl' logs/$run.log | awk '{print $3}' | tr '\n' ' ')"
  ck=$(ls models/checkpoints/$run/step_*.pt 2>/dev/null | tail -1)
  "$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 | sed "s/^/  recall /" >> "$LOG"
  [ -n "$ck" ] && "$PY" temp/haystack_recall_eval.py $run=$ck 2>&1 | grep -E "^$run" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> "$LOG"
done
say "queue DONE"
