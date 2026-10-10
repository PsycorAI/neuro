#!/bin/bash
# queue_after_night_1008.sh -- 2026-10-08 19:45 (Neuro 53). 5080: after r26m -> RESUME r26k (from step 9000).
# 5090: after r26l's diagnostics -> r26n (RC2 + xchunk dose-down). Waits on log lines / trainer exits (no self-match).
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
post(){ g=$1; r=$2; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=$g bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=$g $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  CUDA_VISIBLE_DEVICES=$g $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG; }
(
  until grep -q 'r26m_13L_RC2_hola256_150m\] evals + diagnostics DONE' $LOG; do sleep 120; done
  r=r26k_13L_RC2_hola_seed1_150m
  GPU=1 FREE_MIN=1 UNTIL="2026-10-09 17:55" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] && post 1 $r
) &
(
  until grep -q 'r26l_13L_RC2_conv_sc20frz_120M_150m\] energy-v2 + diagnostics DONE' $LOG; do sleep 120; done
  r=r26n_13L_RC2_xlate_150m
  GPU=0 bash temp/run_gap_queue.sh $r
  post 0 $r
) &
wait
