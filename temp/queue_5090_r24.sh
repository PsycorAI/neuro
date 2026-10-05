#!/bin/bash
# queue_5090_r24.sh <r21g train PID> -- 2026-10-04 (Glenn OK: drop the r21d/e/f transformer baselines, focus on the
# v4 memory question). The old queue (run_gap_queue.sh) was killed by PID while r21g trains; this waits for r21g to
# finish, runs its recall + haystack evals (same format), then: r21i (control) -> r24a (SWAX) -> r24b (stateful) -> r21b.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_queue.log
while kill -0 "$1" 2>/dev/null; do sleep 30; done
run=r21g_w32x1early_150m
echo "[q0 $(date '+%m-%d %H:%M:%S')] [$run] exit=? (trainer finished) :: $(grep 'eval\] val_ppl' logs/$run.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
ck=$(ls models/checkpoints/$run/step_*.pt 2>/dev/null | tail -1)
"$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 | sed "s/^/  recall /" >> $LOG
[ -n "$ck" ] && "$PY" temp/haystack_recall_eval.py $run=$ck 2>&1 | grep -E "^$run" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=0 bash temp/run_gap_queue.sh r21i_x1early_150m r24a_swax32_x1e_150m r24b_stateful_x1e_150m r21b_x1_150m
