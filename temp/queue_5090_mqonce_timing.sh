#!/bin/bash
# queue_5090_mqonce_timing.sh -- 2026-10-03: retry of the 1B mqar_once_per_step timing (failed 16:02 with "CUDA driver
# error: device not ready", right after a B16 run hit the same error -- likely WDDM memory pressure). Runs after the
# r20 A/B ('mqonce DONE'); also re-times the B8 GA8 baseline in the same session for a clean comparison.
cd /home/glenn/projects/neuro
until grep -q 'mqonce DONE' logs/night_5090.log; do sleep 60; done
sleep 20
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/night_5090.log
say(){ echo "[mt $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 9>logs/gap_queue_gpu0.lock
flock 9
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
prof(){
  local tag=$1; shift
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/probe_g4_r19d.yaml --tag $tag --batch 8 --grad_accum 8 --noprof --wait 6 \
    --active 10 --set compile=true compile_dynamic=false "$@" > logs/prof_$tag.log 2>&1
  say "$tag: exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
prof wc_r19d_base2
prof wc_r19d_mqonce2 mqar_once_per_step=true
say "mqonce timing DONE"
