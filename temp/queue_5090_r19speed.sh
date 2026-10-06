#!/bin/bash
# queue_5090_r19speed.sh -- 2026-10-03: 1B timing after the fixed compile_regional (it compiled nothing before) and
# for the r19d sparse-MLP v4 candidate. Waits (blocking) for the 5090 screen queue's gpu0 lock, so it starts when
# the r19 screens finish. Same protocol as night2 (B8 GA8, expandable_segments). Reference (10-03 00:15, whole-model
# compile): c0-S1 10,391 tok/s, tfmr-S1 16,746.
# WATCH: tail -f /home/glenn/projects/neuro/logs/night_5090.log
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/night_5090.log
say(){ echo "[sp $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 9>logs/gap_queue_gpu0.lock
flock 9
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say "r19speed: gpu0 lock held"
prof(){
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  TORCH_LOGS=recompiles $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag "$@" > logs/prof_$tag.log 2>&1
  say "$tag: exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1) | recompiles $(grep -c 'Recompiling' logs/prof_$tag.log)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
G="--batch 8 --grad_accum 8 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false"
prof rcfix_c0s1 probe_g4_c0s1 $G compile_regional=true
prof wc_r19d probe_g4_r19d $G
prof rcfix_r19d probe_g4_r19d $G compile_regional=true
say "r19speed DONE"
