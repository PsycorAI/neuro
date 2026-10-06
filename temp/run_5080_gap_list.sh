#!/bin/bash
# run_5080_gap_list.sh -- 2026-09-28: run the given screen configs on the 5080 (free until ~13:00 Mon; nothing starts after 10:15).
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_screen.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_screen.log
exec 9>logs/gap_screen.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[gap $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for run in "$@"; do
  if [ "$(date +%H%M)" -gt "${DEADLINE:-1015}" ]; then say "[$run] SKIPPED (past start deadline ${DEADLINE:-1015})"; continue; fi
  rm -rf models/checkpoints/$run
  say "[$run] START"
  "$PY" src/train_gpu.py --config configs/$run.yaml --device cuda > logs/$run.log 2>&1
  say "[$run] exit=$? :: $(grep 'eval\] val_ppl' logs/$run.log | awk '{print $3}' | tr '\n' ' ')"
  "$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 >> "$LOG"
done
say "=== 5080 list DONE ==="
