#!/bin/bash
# 2026-09-30 eval-only: r9 primary metrics (streaming ppl by position, memory half-life/pathways). Usage: bash temp/run_eval_r9.sh <run>
cd /home/glenn/projects/neuro
R=$1
K=models/checkpoints/$R/step_0006104.pt
PY=/home/glenn/projects/bdh/venv/bin/python
CUDA_VISIBLE_DEVICES=1 $PY temp/stream_eval.py $R=$K 2>&1 | grep -v -i warn | tee -a logs/stream_eval_baseline.log
CUDA_VISIBLE_DEVICES=1 $PY temp/diag_haystack_pathways.py $R=$K 2>&1 | grep -v -i warn | tee -a logs/diag_haystack_pathways.log
