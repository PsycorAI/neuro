#!/bin/bash
# queue_5090_night.sh -- 2026-10-02 (Glenn OK): after the B4 debug (lock order):
#  (1) graphs prototype PARITY at 30M (all 4 modes in one process; not timing-sensitive)
#  (2) wait until >= 00:30 and the 5080 idle (Glenn gaming), cap 3 h
#  (3) VRAM re-check at GA8: expandable_segments (WDDM cliff at ~30 GB reserved) for c0-S1 and tfmr-S1
#  (4) graphs prototype TIMING at 1B, one mode per process (graph pools would inflate later modes' VRAM)
# WATCH: tail -f /home/glenn/projects/neuro/logs/night_5090.log
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/night_5090.log
say(){ echo "[nt $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 8>logs/5090_seq.lock
flock 8
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
say "lock held"
# (1)
$PY temp/proto_graphs.py --config configs/screen_r16_c0_s0_30m.yaml --batch 32 --iters 10 > logs/proto_graphs_30m.log 2>&1
say "parity 30M exit=$?"; grep '^\[graphs\]' logs/proto_graphs_30m.log | while read -r l; do say "$l"; done
# (2)
n=0
while { [ "$(date +%H%M)" -gt 1200 ] || [ "$(date +%H%M)" -lt 0030 ]; } && [ $n -lt 720 ]; do sleep 30; n=$((n+1)); done
m=0
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 1500 ] && [ $m -lt 360 ]; do sleep 30; m=$((m+1)); done
say "quiet check: 5080 memory $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1); starting timing"
prof(){
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag "$@" > logs/prof_$tag.log 2>&1
  say "$tag: exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
# (3)
G="--batch 8 --grad_accum 8 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True prof g4ga8x_c0s1 probe_g4_c0s1 $G
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True prof g4ga8x_tfmr probe_g4_tfmrs1 $G
prof g4ga8b6_c0s1 probe_g4_c0s1 --batch 6 --grad_accum 8 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false
# (4)
for md in eager graph rc rcgraph; do
  $PY temp/proto_graphs.py --config configs/probe_g4_c0s1.yaml --batch 8 --iters 20 --modes $md > logs/proto_graphs_1b_$md.log 2>&1
  say "proto 1B $md exit=$? $(grep '^\[graphs\]' logs/proto_graphs_1b_$md.log)"
done
say "NIGHT DONE"
