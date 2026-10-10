#!/bin/bash
# queue_5080_r26i.sh -- 2026-10-08 03:50 (Neuro 53). RC2 + stochastic chunk (12L) on the 16 GB 5080 with expandable
# segments (the r26c layout only ALLOCATES ~3.4 GB; its 17 GB reserve is fragmentation). Ends by 17:55 (Glenn).
cd /home/glenn/projects/neuro
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True T=2048 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
r=r26i_13L_RC2_hola_rand128_150m
GPU=1 FREE_MIN=1 UNTIL="2026-10-08 17:55" bash temp/run_5080_sched.sh $r
[ -f models/checkpoints/$r/DONE ] || exit 0
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
for s in xcopy_sweep_eval mem_write_probe multichunk_eval; do CUDA_VISIBLE_DEVICES=1 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
CUDA_VISIBLE_DEVICES=1 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
