#!/bin/bash
# queue_5090_night2.sh -- 2026-10-03 (Glenn OK; replaces queue_5090_night.sh steps 3-4, the 30M parity already ran):
#  (A) clean timing (5080 idle, host quiet): VRAM re-check at GA8, regional-compile check, 1B graphs prototype
#  (B) then start the 5080 overnight screens (after timing, so they can't load the CPU during it)
#  (C) then 5090 overnight screens (train + recall + haystack)
# WATCH: tail -f /home/glenn/projects/neuro/logs/night_5090.log
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/night_5090.log
say(){ echo "[nt $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 8>logs/5090_seq.lock
flock 8
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
say "night2: lock held; 5080 memory $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1)"
prof(){
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag "$@" > logs/prof_$tag.log 2>&1
  say "$tag: exit=$? $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
# (A) timing
G="--batch 8 --grad_accum 8 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True prof g4ga8x_c0s1 probe_g4_c0s1 $G
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True prof g4ga8x_tfmr probe_g4_tfmrs1 $G
prof g4ga8b6_c0s1 probe_g4_c0s1 --batch 6 --grad_accum 8 --noprof --wait 6 --active 10 --set compile=true compile_dynamic=false
prof g4ga8rc_c0s1 probe_g4_c0s1 $G compile_regional=true
for md in eager graph rc rcgraph; do
  $PY temp/proto_graphs.py --config configs/probe_g4_c0s1.yaml --batch 8 --iters 20 --modes $md > logs/proto_graphs_1b_$md.log 2>&1
  say "proto 1B $md exit=$? $(grep '^\[graphs\]' logs/proto_graphs_1b_$md.log)"
done
say "timing DONE"
# (B) 5080 overnight screens (its scheduler pauses at 18:55 and checks the GPU is free first)
GPU=1 setsid -f bash temp/run_5080_sched.sh screen_r17f_k688_s1_30m screen_r17k_k688_ramp03_s0_30m \
  screen_r17g_ramp03_s1_30m > /dev/null 2>&1
say "5080 scheduler launched"
# (C) 5090 overnight screens
GPU=0 bash temp/run_gap_queue.sh screen_r17i_k512_ramp04_s1_30m screen_r17j_x2_target20_s1_30m \
  screen_r17k_k688_ramp03_s1_30m
say "NIGHT2 DONE"
