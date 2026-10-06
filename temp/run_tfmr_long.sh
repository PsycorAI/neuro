#!/bin/bash
# run_tfmr_long.sh -- 2026-09-26: induction-head hypothesis. The 30M fixed transformer
# (AdamW) sits at chance on MQAR after 100M tok. Same config at 600M tok (~1.4h @ ~120k
# tok/s): if recall emerges, the 100M-token screens are just too short for attention.
# 5080 must be free for Glenn ~09:20 -> resumable; kill with pkill -f run_tfmr_long.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/tfmrfix_screen.log
say(){ echo "[tfmrfix $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
sed -e 's/^run_name: .*/run_name: screen_tfmrfix_adamw600m_s0_30m/' -e 's/^max_tokens: .*/max_tokens: 600000000/' \
    configs/screen_tfmrfix_adamw_s0_30m.yaml > configs/screen_tfmrfix_adamw600m_s0_30m.yaml
say "START screen_tfmrfix_adamw600m_s0_30m (600M tok)"
$PY src/train_gpu.py --config configs/screen_tfmrfix_adamw600m_s0_30m.yaml --device cuda >> "$LOG" 2>&1
say "train exit=$?"
$PY scripts/recall_eval.py --runs screen_tfmrfix_adamw_s0_30m,screen_tfmrfix_adamw600m_s0_30m --B 32 >> "$LOG" 2>&1
say "DONE (600M)"
