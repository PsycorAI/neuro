#!/bin/bash
# queue_5090_r17.sh -- 2026-10-02 (Glenn OK): on the 5090, holding logs/5090_seq.lock (so the queued 1B neuron probe
# waits): (1) throughput attribution at the 30M screen shape -- clean timing (--noprof) and op profiles of c0 / x2 /
# l3 / t4, plus exact-math speed variants of c0; (2) the r17 DSN config screens.
# WATCH: tail -f /home/glenn/projects/neuro/logs/r17_5090.log
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/r17_5090.log
say(){ echo "[r17 $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
exec 8>logs/5090_seq.lock
say "waiting for logs/5090_seq.lock (v3 post-training chain must end first)"
flock 8
say "lock held; GPU0 free"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
prof(){   # prof <tag> <config> [profile-args...]
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag --batch 32 "$@" > logs/prof_$tag.log 2>&1
  say "$tag: $(grep -E '^\[prof' logs/prof_$tag.log | head -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
# (1a) clean timing, profiler off
prof t_c0 screen_r16_c0_s0_30m --noprof --wait 15 --active 30
prof t_x2 screen_r16_x2_dsnlif_s0_30m --noprof --wait 15 --active 30
prof t_l3 screen_r16_l3_gelu_s0_30m --noprof --wait 15 --active 30
prof t_t4 screen_r16_t4_tfmr4_fair_s0_30m --noprof --wait 15 --active 30
prof t_c0_ch256 screen_r16_c0_s0_30m --noprof --wait 15 --active 30 --set gdn2_chunk_size=256 gdn2_ckpt_chunks=false
prof t_c0_flash screen_r16_c0_s0_30m --noprof --wait 15 --active 30 --set attn_topk=0
prof t_c0_compile screen_r16_c0_s0_30m --noprof --wait 25 --active 30 --set compile=true
prof t_t4_compile screen_r16_t4_tfmr4_fair_s0_30m --noprof --wait 25 --active 30 --set compile=true
prof t_c0_b64 screen_r16_c0_s0_30m --noprof --wait 15 --active 30 --batch 64
# (1b) op profiles (where the time goes)
prof p_c0 screen_r16_c0_s0_30m --wait 10 --active 6
prof p_t4 screen_r16_t4_tfmr4_fair_s0_30m --wait 10 --active 6
say "profiles done"
# (2) r17 DSN config screens (train + recall + haystack per run)
GPU=0 bash temp/run_gap_queue.sh screen_r17f_k688_s0_30m screen_r17g_ramp03_s0_30m screen_r17i_k512_ramp04_s0_30m \
  screen_r17h_wide4096_s0_30m screen_r17j_x2_target20_s0_30m
say "DONE"
