#!/bin/bash
# queue_5080_r23.sh -- 2026-10-04 22:50. Glenn lent the 5080 from now until 18:00 10-05.
# r23a evals (its trainer finished 18:28; the old scheduler was sleeping to 00:00), then r23b (6L depth proxy).
# Launch: setsid -f bash temp/queue_5080_r23.sh > /dev/null 2>&1
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_queue.log
run=screen_r23a_g3w_4L_x1e; last=$(ls models/checkpoints/$run/step_*.pt | tail -1)
echo "[s1 $(date '+%m-%d %H:%M:%S')] [$run] evals ($last)" >> $LOG
"$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 | sed "s/^/  recall /" >> $LOG
"$PY" temp/haystack_recall_eval.py $run=$last 2>&1 | grep -E "^$run" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=1 UNTIL="2026-10-05 18:00" exec bash temp/run_5080_sched.sh screen_r23b_g3w_6L_x1e
