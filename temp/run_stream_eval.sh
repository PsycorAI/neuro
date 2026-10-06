#!/bin/bash
# 2026-09-29 eval-only: carried-state vs zero-state on random-window-trained ckpts (GPU 1, beside training)
cd /home/glenn/projects/neuro
K=models/checkpoints
CUDA_VISIBLE_DEVICES=1 /home/glenn/projects/bdh/venv/bin/python temp/stream_eval.py \
  lif_exact=$K/screen_honest_lifexact_s0_30m/step_0006104.pt lev4=$K/screen_gap_lev4_s0_30m/step_0006104.pt \
  hybrid=$K/screen_honest_hybrid_s0_30m/step_0006104.pt full_stack=$K/screen_gap_hyb_full_s0_30m/step_0006104.pt \
  full+gap440=$K/screen_r6_gap_s0_30m/step_0006104.pt 2>&1 | grep -v -i warn | tee logs/stream_eval_baseline.log
