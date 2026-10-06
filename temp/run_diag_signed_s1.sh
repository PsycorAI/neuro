#!/bin/bash
# 2026-09-30 eval-only: streaming + memory half-life on signed seed 1 (r9 control, 2nd seed)
cd /home/glenn/projects/neuro
K=models/checkpoints/screen_r8_signed_s1_30m/step_0006104.pt
PY=/home/glenn/projects/bdh/venv/bin/python
CUDA_VISIBLE_DEVICES=1 $PY temp/diag_haystack_pathways.py signed_s1=$K 2>&1 | grep -v -i warn | grep ' full ' | tee -a logs/diag_haystack_pathways.log
CUDA_VISIBLE_DEVICES=1 $PY temp/stream_eval.py signed_s1=$K 2>&1 | grep -v -i warn | tee -a logs/stream_eval_baseline.log
grep '\[spike\]' logs/screen_r8_signed_s1_30m.log | tail -1
