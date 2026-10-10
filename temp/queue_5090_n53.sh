#!/bin/bash
# queue_5090_n53.sh -- 2026-10-06 23:00 (Neuro 53; Glenn: keep both GPUs busy overnight). Starts after Neuro 52's
# queue_5090_r25z.sh (r25z way 2 + its evals) has finished. r25x conversion, then r26c (chunk 1024 / T2048).
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python
while pgrep -f "bash temp/queue_5090_r25z.sh" > /dev/null; do sleep 60; done
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5090 queue (r25x, r26c) starting" >> $LOG
SCALE=1.0 bash temp/queue_spk_conv.sh 0 r25x_13L_spk_kd_sc10_150m models/checkpoints/r25o_13L_H6only_150m/step_0012208.pt
r=r26c_13L_H6_chunk1024_150m
GPU=0 bash temp/run_gap_queue.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=0 T=2048 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval.py "$r=$ck" 2>&1 | grep energy | sed "s/^/  energy /" >> $LOG
for s in xcopy_sweep_eval mem_write_probe; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5090 queue (r25x, r26c) DONE" >> $LOG
