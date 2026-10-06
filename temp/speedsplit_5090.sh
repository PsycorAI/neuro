#!/bin/bash
# speedsplit_5090.sh -- 2026-10-02 (Glenn OK): clean-GPU timing to split the c0-vs-t4 speed gap into
# optimizer (unbatched NorMuon) vs per-step GDN-2 telemetry vs the H0 layer itself, plus a compile diagnosis.
# Runs on GPU0 while the r17 queue runner (PID 276875) is frozen (kill -STOP); waits for r17g training (PID 279145)
# to exit, then resumes the runner (kill -CONT) at the end.
# WATCH: tail -f /home/glenn/projects/neuro/logs/speedsplit_5090.log
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/speedsplit_5090.log
say(){ echo "[ss $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
say "waiting for r17g training (PID 279145) to exit"
# (kill -0 succeeds on a zombie: the frozen runner can't reap it -> check the process state instead)
while ps -o stat= -p 279145 2>/dev/null | grep -qv Z; do sleep 10; done
say "GPU0 free; starting"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
prof(){   # prof <tag> <config> [profile-args...]
  local tag=$1 cfg=$2; shift 2
  rm -rf models/checkpoints/screen_prof_$tag
  $PY temp/profile_train.py --config configs/$cfg.yaml --tag $tag --batch 32 "$@" > logs/prof_$tag.log 2>&1
  say "$tag: $(grep -E '^\[prof' logs/prof_$tag.log | head -1)"
  rm -rf models/checkpoints/screen_prof_$tag temp/prof_$tag.yaml
}
C0=screen_r16_c0_s0_30m
T4=screen_r16_t4_tfmr4_fair_s0_30m
A="--noprof --wait 15 --active 30"
prof s_c0 $C0 $A
prof s_t4 $T4 $A
prof s_c0_bns $C0 $A --set muon_batched_ns=true
prof s_c0_tel $C0 $A --set telemetry_eval_only=true
prof s_c0_adamw $C0 $A --set optimizer=adamw
prof s_c0_bns_tel_ch256 $C0 $A --set muon_batched_ns=true telemetry_eval_only=true gdn2_chunk_size=256 gdn2_ckpt_chunks=false
# compile diagnosis: why compiled c0 is 20.7 s/step
TORCH_LOGS=recompiles,graph_breaks prof s_c0_compile_diag $C0 --noprof --wait 5 --active 3 --set compile=true
prof s_c0_compile_tel $C0 --noprof --wait 25 --active 30 --set compile=true telemetry_eval_only=true
say "done; resuming r17 queue runner"
kill -CONT 276875
say "resumed"
