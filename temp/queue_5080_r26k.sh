#!/bin/bash
# queue_5080_r26k.sh -- 2026-10-08 12:55 (Neuro 53). RC2 seed 1 on the 5080; pauses 17:50 (Glenn's evening), resumes 00:00.
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
r=r26k_13L_RC2_hola_seed1_150m
GPU=1 FREE_MIN=1 PAUSE=1750 NOSTART=1730 bash temp/run_5080_sched.sh $r
[ -f models/checkpoints/$r/DONE ] || exit 0
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=1 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
CUDA_VISIBLE_DEVICES=1 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
