#!/bin/bash
# resume of run_5080_honest_s1.sh after the runner was SIGTERM'd 2026-09-27 ~21:4x: finish fpt_s1 (auto-resumes, NO rm), then recall.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/honest_screen.log
say(){ echo "[honest $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
r=screen_honest_fpt_s1_30m
say "[$r] RESUME"
$PY src/train_gpu.py --config configs/$r.yaml --device cuda >> logs/$r.log 2>&1
say "[$r] exit=$? :: $(grep 'eval\]' logs/$r.log | awk '{print $3}' | tr '\n' ' ')"
$PY scripts/recall_eval.py --runs screen_honest_fpt_s0_30m,screen_honest_fpt_s1_30m,screen_honest_lifexact_s0_30m,screen_honest_lifexact_s1_30m --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E '^ *(run|screen|chance)' >> "$LOG"
say "DONE"
