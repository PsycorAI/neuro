#!/bin/bash
# run_5080_tfmrfix.sh -- 2026-09-26: 30M FIXED-transformer preview on the 5080, queued
# behind the depth screen (waits on its flock). Control = screen_bs512_s0_30m (done).
# Recall = SINGLE-PASS for both (split replay has no KV carry -> invalid for attention).
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1          # 5080 ONLY
export PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/tfmrfix_screen.log
say(){ echo "[tfmrfix $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "queued: waiting for depth screen lock"
flock logs/depth_screen.lock -c true
say "START screen_tfmrfix_s0_30m"
"$PY" src/train_gpu.py --config configs/screen_tfmrfix_s0_30m.yaml --device cuda >> "$LOG" 2>&1
say "train exit=$?"
"$PY" scripts/recall_eval.py --runs screen_bs512_s0_30m,screen_tfmrfix_s0_30m --Ns 4,8,16,32,64 --B 32 >> "$LOG" 2>&1
say "recall exit=$?"
say "DONE"
