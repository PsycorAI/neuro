#!/bin/bash
# queue_5090_mqonce.sh -- 2026-10-03: after queue_5090_mqarprof.sh ('mqarprof DONE'): (1) 1B timing of
# mqar_once_per_step (r19d probe, B8 GA8); (2) 30M quality A/B r20a (GA4 control) vs r20b (GA4 + mqar_once_per_step).
cd /home/glenn/projects/neuro
until grep -q 'mqarprof DONE' logs/night_5090.log; do sleep 60; done
sleep 20
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/night_5090.log
say(){ echo "[mo $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
(
  exec 9>logs/gap_queue_gpu0.lock
  flock 9
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  tag=wc_r19d_mqonce
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/probe_g4_r19d.yaml --tag $tag --batch 8 --grad_accum 8 --noprof --wait 6 \
    --active 10 --set compile=true compile_dynamic=false mqar_once_per_step=true > logs/prof_$tag.log 2>&1
  say "$tag: exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
)
GPU=0 bash temp/run_gap_queue.sh screen_r20a_ga4ctrl_s0_30m screen_r20b_ga4mqonce_s0_30m
say "mqonce DONE"
