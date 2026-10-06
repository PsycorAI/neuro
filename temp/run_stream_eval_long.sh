#!/bin/bash
# 2026-09-30 eval-only: 1M-token carried-state stability, stateful-trained vs random-trained (GPU 1, beside training)
cd /home/glenn/projects/neuro
K=models/checkpoints
CUDA_VISIBLE_DEVICES=1 /home/glenn/projects/bdh/venv/bin/python temp/stream_eval_long.py \
  signed_s0=$K/screen_r5_signed_s0_30m/step_0006104.pt signed_s1=$K/screen_r8_signed_s1_30m/step_0006104.pt \
  r9a_state=$K/screen_r9a_stateful_s0_30m/step_0006104.pt r9c_st_swa=$K/screen_r9c_stateful_swa_s0_30m/step_0006104.pt \
  full_stack=$K/screen_gap_hyb_full_s0_30m/step_0006104.pt full+gap440=$K/screen_r6_gap_s0_30m/step_0006104.pt \
  2>&1 | grep -v -i warn | tee logs/stream_eval_long.log
