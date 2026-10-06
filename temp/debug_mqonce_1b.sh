#!/bin/bash
# debug_mqonce_1b.sh -- 2026-10-03: why does the 1B r19d probe die ("CUDA driver error: device not ready") in the
# MQAR forward with mqar_once_per_step (MQAR batch 8 x GA8 = 64)? (a) eager (compile off); (b) compiled with
# CUDA_LAUNCH_BLOCKING=1 for the real failing kernel; (c) compiled, MQAR batch 32 (mqar_batch 4 x GA8).
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>logs/gap_queue_gpu0.lock
flock 9
run(){
  local tag=$1; shift
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/probe_g4_r19d.yaml --tag $tag --batch 8 --grad_accum 8 --noprof --wait 4 \
    --active 6 --set "$@" > logs/prof_$tag.log 2>&1
  echo "[dbg $(date +%H:%M:%S)] $tag exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) $(grep -E 'Error' logs/prof_$tag.log | grep -v Warning | tail -1 | cut -c1-160)" >> logs/night_5090.log
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
run dbg_eager compile=false mqar_once_per_step=true
CUDA_LAUNCH_BLOCKING=1 run dbg_clb compile=true compile_dynamic=false mqar_once_per_step=true
echo "[dbg $(date +%H:%M:%S)] debug DONE" >> logs/night_5090.log
