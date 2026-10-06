#!/bin/bash
# queue_5090_r21_reorder.sh <r21c train PID> -- 2026-10-04: the original r21 queue was stopped while r21c trains (so the
# order can change); this waits for r21c to finish, runs its recall + haystack evals (same format as run_gap_queue.sh),
# then runs the reordered rest: decision-relevant arms first, transformer baselines last.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_queue.log
while kill -0 "$1" 2>/dev/null; do sleep 30; done
run=r21c_w128x1_150m
echo "[q0 $(date '+%m-%d %H:%M:%S')] [$run] exit=? (trainer finished) :: $(grep 'eval\] val_ppl' logs/$run.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
ck=$(ls models/checkpoints/$run/step_*.pt 2>/dev/null | tail -1)
"$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 | sed "s/^/  recall /" >> $LOG
[ -n "$ck" ] && "$PY" temp/haystack_recall_eval.py $run=$ck 2>&1 | grep -E "^$run" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=0 bash temp/run_gap_queue.sh r21b_x1_150m r21g_w32x1early_150m r21i_x1early_150m r21e_tfmr_w128x1_150m r21d_tfmr_x1_150m r21f_w128_150m
