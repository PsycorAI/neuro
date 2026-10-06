#!/bin/bash
# queue_5090_g4.sh -- 2026-10-02 (Glenn OK: speed plan B + C). Waits for logs/5090_seq.lock (the r17 queue), then on GPU0:
#  (1) 30M compile test of the fix (mqar_pad_to + _kwta helper + standard speed flags): eager vs compiled c0, recompile count
#  (2) G4 1B fit sweep (VRAM + rough tok/s; the 5080 job may still share the CPU) -- also warms the inductor cache
#  (3) waits until the 5080 is idle (its scheduler stops at 18:55) so the host CPU is quiet
#  (4) G4 clean timing (c0-S1 vs param-matched transformer) + op profile of c0-S1 (GPU-busy fraction)
# Configs: temp/mk_g4_configs.py. Plan: docs/speed_plan_v4_2026_10_02.md.
# WATCH: tail -f /home/glenn/projects/neuro/logs/g4_5090.log
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/g4_5090.log
say(){ echo "[g4 $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 8>logs/5090_seq.lock
say "waiting for logs/5090_seq.lock (r17 queue)"
flock 8
say "lock held; GPU0 free"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
prof(){   # prof <tag> <config> [profile-args...]
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag "$@" > logs/prof_$tag.log 2>&1
  local rc=$?
  say "$tag: exit=$rc $(grep -E '^\[prof' logs/prof_$tag.log | head -1) | $(grep -o 'vram [0-9.]*/[0-9.]* GB' logs/prof_$tag.log | tail -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
C0=screen_r16_c0_s0_30m
STD="gdn2_chunk_size=256 gdn2_ckpt_chunks=false telemetry_eval_only=true muon_batched_ns=true mqar_pad_to=65"
# (1) 30M compile test (the k-ramp starts at 60% of the short run, so compiled timing also covers changing k)
#done 17:08 prof c2_c0_eager_std $C0 --batch 32 --noprof --wait 25 --active 30 --set $STD
#done 17:15 TORCH_LOGS=recompiles,graph_breaks prof c2_c0_compile_std $C0 --batch 32 --noprof --wait 25 --active 30 --set $STD compile=true
#say "c2_c0_compile_std: recompiles=$(grep -c 'Recompiling function' logs/prof_c2_c0_compile_std.log) graph_breaks=$(grep -c 'Graph break in user code' logs/prof_c2_c0_compile_std.log)"
# (2) G4 fit sweep
G="--noprof --wait 12 --active 20 --set compile=true compile_dynamic=false"
prof g4fit_c0s1_b4 probe_g4_c0s1 --batch 4 $G
prof g4fit_c0s1_b8 probe_g4_c0s1 --batch 8 $G
prof g4fit_tfmr_b8 probe_g4_tfmrs1 --batch 8 $G
prof g4fit_tfmr_b16 probe_g4_tfmrs1 --batch 16 $G
# (3) quiet host: wait for the 5080 to go idle
say "waiting for the 5080 to go idle"
n=0   # cap the wait at 2 h (Glenn may use the 5080 himself after 19:00)
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 1500 ] && [ $n -lt 240 ]; do sleep 30; n=$((n+1)); done
say "5080 memory $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1) after ${n} x 30 s; clean timing"
# (4) G4 clean timing + profile
prof g4_c0s1_b8 probe_g4_c0s1 --batch 8 $G
prof g4_c0s1_b4 probe_g4_c0s1 --batch 4 $G
prof g4_c0s1_b8_eager probe_g4_c0s1 --batch 8 --noprof --wait 8 --active 15
prof g4_tfmr_b8 probe_g4_tfmrs1 --batch 8 $G
prof g4_tfmr_b16 probe_g4_tfmrs1 --batch 16 $G
prof g4p_c0s1_b8 probe_g4_c0s1 --batch 8 --wait 12 --active 6 --set compile=true compile_dynamic=false
prof g4p_tfmr_b8 probe_g4_tfmrs1 --batch 8 --wait 12 --active 6 --set compile=true compile_dynamic=false
say "DONE"
