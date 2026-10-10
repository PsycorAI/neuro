#!/bin/bash
# queue_5090_r26s.sh -- 2026-10-07 07:35 (Neuro 53). After queue_5090_n53.sh (r25x, r26c): r26s = r25o seed 1 + evals.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python
while pgrep -f "bash temp/queue_5090_n53.sh" > /dev/null; do sleep 60; done
r=r26s_13L_H6_seed1_150m
GPU=0 bash temp/run_gap_queue.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
for s in xcopy_sweep_eval mem_write_probe multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26s queue DONE" >> $LOG
