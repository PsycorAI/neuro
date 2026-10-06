#!/bin/bash
# run_5090_gap_stream.sh -- 2026-09-28 ppl-gap plan B/C/D arms on the 5090 (Glenn: 5090 free, all code arms OK).
# Usage: run_5090_gap_stream.sh <stream-tag> <run> [<run> ...]   (3 streams run concurrently; launch-bound 30M)
# Auto-kill: eval2 > 170 (control eval2 mean 154 x 1.10). Recall = single-pass MQAR N4..64 after each finished arm.
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_5090.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_5090.log
TAG=$1; shift
exec 9>logs/gap_5090_$TAG.lock; flock -n 9 || { echo "stream $TAG already running"; exit 1; }
say(){ echo "[5090-$TAG $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for run in "$@"; do
  alog=logs/$run.log
  rm -rf models/checkpoints/$run
  say "[$run] START"
  "$PY" src/train_gpu.py --config configs/$run.yaml --device cuda > "$alog" 2>&1 &
  pid=$!; killed=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    n=$(grep -c 'eval\] val_ppl' "$alog")
    v=$(grep 'eval\] val_ppl' "$alog" | tail -1 | awk '{print $3}')
    if [ "$n" -eq 2 ] && awk "BEGIN{exit !($v > 170)}"; then
      say "[$run] KILL eval2 $v > 170"; kill -INT $pid; wait $pid; killed=1; break
    fi
  done
  [ $killed -eq 1 ] && continue
  wait $pid; say "[$run] exit=$? :: $(grep 'eval\] val_ppl' "$alog" | awk '{print $3}' | tr '\n' ' ')"
  "$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 | sed "s/^/[5090-$TAG recall] /" >> "$LOG"
done
say "stream DONE"
