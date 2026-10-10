#!/bin/bash
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
r=r26k_13L_RC2_hola_seed1_150m
while pgrep -f "configs/$r.yaml" > /dev/null; do sleep 60; done
echo "[s1 $(date '+%m-%d %H:%M:%S')] [$r] exit :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
grep -q '^done:' logs/$r.log || { echo "[n53] [$r] ENDED WITHOUT done:" >> $LOG; exit 1; }
touch models/checkpoints/$r/DONE; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=1 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
CUDA_VISIBLE_DEVICES=1 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
