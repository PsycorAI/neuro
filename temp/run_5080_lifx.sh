#!/bin/bash
# run_5080_lifx.sh -- 2026-09-27 rate-matched exact-neuron 30M screen (seed 0), vs control
# screen_indepwin_s0_30m (val_ppl 95.74 @1k, 79.30 @2k, 54.34 final; recall split N32/N64 81/44).
# Arms: exact LIF (gamma=1) and no-reset PSN-style (gamma=0), both sparsity_target 0.18.
# Auto-kill: eval#1 > 110 or eval#2 > 87.2 (10% worse than control) -> SIGINT, next arm.
# WATCH: tail -f /home/glenn/projects/neuro/logs/lifx_screen.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/lifx_screen.log
exec 9>logs/lifx_screen.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[lifx $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
arm(){ # $1 = tag
  local run=screen_lifx_$1_s0_30m alog=logs/screen_lifx_$1.log pid n v
  rm -rf models/checkpoints/$run
  say "[$1] START"
  "$PY" src/train_gpu.py --config configs/$run.yaml --device cuda > "$alog" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    n=$(grep -c 'eval\] val_ppl' "$alog")
    v=$(grep 'eval\] val_ppl' "$alog" | tail -1 | awk '{print $3}')
    if [ "$n" -eq 1 ] && awk "BEGIN{exit !($v > 110)}"; then say "[$1] KILL eval1 $v > 110"; kill -INT $pid; wait $pid; return; fi
    if [ "$n" -eq 2 ] && awk "BEGIN{exit !($v > 87.2)}"; then say "[$1] KILL eval2 $v > 87.2"; kill -INT $pid; wait $pid; return; fi
  done
  wait $pid; say "[$1] exit=$? :: $(grep 'eval\]' "$alog" | tr '\n' ' ' | tr -s ' ')"
  "$PY" scripts/recall_eval.py --runs screen_indepwin_s0_30m,$run --Ns 4,8,16,32,64 --B 32 --replay split 2>&1 | grep -E '^ *(run|screen|chance)' >> "$LOG"
}
arm g1_t18
arm g0_t18
say "DONE"
arm fptK3
arm fptK1
say "DONE (K arms)"
arm fptK0
say "DONE (K0)"
arm fptK2
arm fptK4
say "DONE (even K)"
