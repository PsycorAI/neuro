#!/bin/bash
# 2026-09-29 eval-only: memory half-life + streaming eval on the signed-spike full stack; spike-rate comparison
cd /home/glenn/projects/neuro
K=models/checkpoints/screen_r5_signed_s0_30m/step_0006104.pt
PY=/home/glenn/projects/bdh/venv/bin/python
CUDA_VISIBLE_DEVICES=1 $PY temp/diag_haystack_pathways.py signed=$K 2>&1 | grep -v -i warn | tee -a logs/diag_haystack_pathways.log
CUDA_VISIBLE_DEVICES=1 $PY temp/stream_eval.py signed=$K 2>&1 | grep -v -i warn | tee -a logs/stream_eval_baseline.log
for r in screen_gap_hyb_full_s0_30m screen_r5_full_s1_30m screen_r5_signed_s0_30m screen_honest_lifexact_s0_30m; do
  echo "$r: $(grep '\[spike\]' logs/$r.log | tail -1)"
done
