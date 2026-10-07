#!/bin/bash
# queue_5090_r25z.sh -- 2026-10-06 20:00. After queue_5090_r25y.sh (r25y head 512): way 2 = r25z (14L, memory at 4 + 9).
cd /home/glenn/projects/neuro
while pgrep -f "bash temp/queue_5090_r25y.sh" > /dev/null; do sleep 60; done
r=r25z_14L_H4H9_150m
GPU=0 bash temp/run_gap_queue.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
CUDA_VISIBLE_DEVICES=0 /home/glenn/projects/bdh/venv/bin/python temp/energy_eval.py "$r=$ck" 2>&1 | grep energy | sed "s/^/  energy /" >> logs/gap_queue.log
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25z queue DONE" >> logs/gap_queue.log
