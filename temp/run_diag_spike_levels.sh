#!/bin/bash
# 2026-09-30 eval-only: spike magnitude vs events, unsigned lev4 vs signed variants (GPU 1, beside training)
cd /home/glenn/projects/neuro
K=models/checkpoints
CUDA_VISIBLE_DEVICES=1 /home/glenn/projects/bdh/venv/bin/python temp/diag_spike_levels.py \
  full_stack=$K/screen_gap_hyb_full_s0_30m/step_0006104.pt signed=$K/screen_r5_signed_s0_30m/step_0006104.pt \
  signed_lam2=$K/screen_r8_signed_lam2_s0_30m/step_0006104.pt signed_canon=$K/screen_r8_signed_canon_s0_30m/step_0006104.pt \
  lif_exact=$K/screen_honest_lifexact_s0_30m/step_0006104.pt 2>&1 | grep -v -i warn | tee logs/diag_spike_levels.log
