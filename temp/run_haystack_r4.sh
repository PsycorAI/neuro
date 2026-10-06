#!/bin/bash
cd /home/glenn/projects/neuro
K=models/checkpoints
CUDA_VISIBLE_DEVICES=1 /home/glenn/projects/bdh/venv/bin/python temp/haystack_recall_eval.py \
  lif_exact=$K/screen_honest_lifexact_s0_30m/step_0006104.pt lev4=$K/screen_gap_lev4_s0_30m/step_0006104.pt \
  hybrid=$K/screen_honest_hybrid_s0_30m/step_0006104.pt full_stack=$K/screen_gap_hyb_full_s0_30m/step_0006104.pt \
  hyb3_lev4=$K/screen_gap_hyb3_lev4_s0_30m/step_0006104.pt tfmr2=$K/screen_tfmrfix_adamw_s0_30m/step_0006104.pt \
  tfmr3=$K/screen_tfmr3_adamw_s0_30m/step_0006104.pt 2>&1 | grep -v Warn | tee logs/haystack_recall_r4.log
