#!/bin/bash
# 2026-09-30 eval-only: inference with a short attention window, r9b vs signed controls (GPU 1, beside training)
cd /home/glenn/projects/neuro
K=models/checkpoints
CUDA_VISIBLE_DEVICES=1 /home/glenn/projects/bdh/venv/bin/python temp/eval_attn_window.py \
  r9b_swa=$K/screen_r9b_swastoch_s0_30m/step_0006104.pt signed_s0=$K/screen_r5_signed_s0_30m/step_0006104.pt \
  signed_s1=$K/screen_r8_signed_s1_30m/step_0006104.pt 2>&1 | grep -v -i warn | tee logs/eval_attn_window.log
