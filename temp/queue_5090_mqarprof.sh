#!/bin/bash
# queue_5090_mqarprof.sh -- 2026-10-03: how much of the 1B step is the MQAR aux pass? (it runs inside the grad-accum
# loop: at GA8 = 8 extra small fwd+bwd per step, launch-bound). Upper bound = the same timing with mqar_aug off.
# Waits for the r19c s2 haystack line (end of queue_5090_r19c_seeds.sh), then times on the free 5090.
cd /home/glenn/projects/neuro
until grep -q 'haystack.*screen_r19c_layer10_s2_30m' logs/gap_queue.log; do sleep 60; done
sleep 20
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/night_5090.log
say(){ echo "[mp $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 9>logs/gap_queue_gpu0.lock
flock 9
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
prof(){
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag "$@" > logs/prof_$tag.log 2>&1
  say "$tag: exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
G="--batch 8 --grad_accum 8 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false"
prof wc_r19d_nomqar probe_g4_r19d $G mqar_aug=false
prof wc_r19d_b16ga4 probe_g4_r19d --batch 16 --grad_accum 4 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false
say "mqarprof DONE"
