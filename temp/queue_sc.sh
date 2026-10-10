#!/bin/bash
# queue_sc.sh -- 2026-10-07 13:25 (Neuro 53). Stochastic chunk size (attn_chunk_short): p26y (7L, 5080, after p26x; must
# start by 17:00) and r26f (12L RC layout, 5090, after the RC queue). Each: r25 evals + E1/E2/E3 diagnostics.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python
diag(){ for s in xcopy_sweep_eval mem_write_probe multichunk_eval; do CUDA_VISIBLE_DEVICES=$1 $PY temp/$s.py "$2=$3" >> logs/e123_diag.log 2>&1; done
        CUDA_VISIBLE_DEVICES=$1 $PY temp/xcopy_sweep_eval.py --contents none "$2=$3" >> logs/e123_diag.log 2>&1; }
(
  while pgrep -f "bash temp/queue_5080_p26x.sh" > /dev/null; do sleep 60; done
  r=screen_p26y_7L_chunk_rand128
  if [ "$(date +%H%M)" -ge 1700 ]; then echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] SKIPPED (late)" >> $LOG; exit 0; fi
  GPU=1 UNTIL="2026-10-07 18:55" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] || exit 0
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  diag 1 $r $ck
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + E1-E3 DONE" >> $LOG
) &
(
  while pgrep -f "bash temp/queue_5090_rc.sh" > /dev/null; do sleep 60; done
  r=r26f_13L_H6_chunk1024_rand128_150m
  GPU=0 bash temp/run_gap_queue.sh $r
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=0 T=2048 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  diag 0 $r $ck
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + E1-E3 DONE" >> $LOG
) &
wait
