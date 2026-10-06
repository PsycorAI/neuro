#!/bin/bash
cd /home/glenn/projects/neuro
r=r25v_13L_H6_xcopy_150m
GPU=0 bash temp/run_gap_queue.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25v queue DONE" >> logs/gap_queue.log
