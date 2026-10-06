#!/bin/bash
# queue_5090_g4ga.sh -- 2026-10-02 (Glenn OK): G4 at grad_accum 8 -- does NorMuon's ~460 ms/step amortize away?
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/g4_5090.log
say(){ echo "[g4 $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 8>logs/5090_seq.lock
flock 8
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
prof(){
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag "$@" > logs/prof_$tag.log 2>&1
  say "$tag: exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
G="--batch 8 --grad_accum 8 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false"
prof g4ga8_c0s1 probe_g4_c0s1 $G
prof g4ga8_tfmr probe_g4_tfmrs1 $G
say "GA8 DONE"
