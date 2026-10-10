#!/bin/bash
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
for r in r26i_13L_RC2_hola_rand128_150m r26e_13L_H6_chunk1024_xlate_150m; do
  GPU=0 FREE_MIN=1 UNTIL="2026-10-10 06:00" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval mem_write_probe multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  CUDA_VISIBLE_DEVICES=0 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
done
