#!/bin/bash
# run_tfmr_adamw.sh -- 2026-09-26: does the 30M fixed transformer learn MQAR with a
# transformer-standard optimizer (AdamW lr 1e-3, wd 0.1) instead of the spiking-tuned
# NorMuon recipe? Everything else = screen_tfmrfix_s0_30m. 5080, ~20 min.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/tfmrfix_screen.log
say(){ echo "[tfmrfix $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
sed -e 's/^run_name: .*/run_name: screen_tfmrfix_adamw_s0_30m/' -e 's/^optimizer: .*/optimizer: adamw/' -e 's/^lr: .*/lr: 0.001/' \
    configs/screen_tfmrfix_s0_30m.yaml > configs/screen_tfmrfix_adamw_s0_30m.yaml
grep -E '^(run_name|optimizer|lr|weight_decay):' configs/screen_tfmrfix_adamw_s0_30m.yaml | tee -a "$LOG"
say "START screen_tfmrfix_adamw_s0_30m"
$PY src/train_gpu.py --config configs/screen_tfmrfix_adamw_s0_30m.yaml --device cuda >> "$LOG" 2>&1
say "train exit=$?"
$PY scripts/recall_eval.py --runs screen_bs512_s0_30m,screen_tfmrfix_s0_30m,screen_tfmrfix_adamw_s0_30m --B 32 >> "$LOG" 2>&1
say "DONE"
