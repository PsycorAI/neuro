#!/bin/bash
# queue_night_1008.sh -- 2026-10-08 13:15 (Neuro 53; Glenn: both GPUs all night). Two independent branches.
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
H=models/checkpoints/r26h_13L_RC_hola64_150m/step_0012208.pt
diag(){ g=$1; shift; for s in "$@"; do CUDA_VISIBLE_DEVICES=$g $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done; }
(   # 5090: after r26i + r26e -> r26g2 (Llama KD conversion) -> r26l (2x frozen conversion)
  while pgrep -f "^bash temp/queue_5090_r26i_e.sh$" > /dev/null; do sleep 60; done
  for rs in "r26g2_13L_RC2_conv_llamakd_150m 1.0" "r26l_13L_RC2_conv_sc20frz_120M_150m 2.0"; do
    set -- $rs; r=$1
    SCALE=$2 bash temp/queue_spk_conv.sh 0 $r $H; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
    CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
    diag 0 xcopy_sweep_eval multichunk_eval
    echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] energy-v2 + diagnostics DONE" >> $LOG
  done
) &
(   # 5080: after r26k -> r26m (HOLA 256)
  while pgrep -f "^bash temp/after_r26k.sh$" > /dev/null; do sleep 60; done
  r=r26m_13L_RC2_hola256_150m
  GPU=1 FREE_MIN=1 UNTIL="2026-10-09 17:55" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] && { ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
    GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
    diag 1 xcopy_sweep_eval mem_write_probe multichunk_eval
    CUDA_VISIBLE_DEVICES=1 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
    echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG; }
) &
wait
echo "[n53 $(date '+%m-%d %H:%M:%S')] night queue 10-08 DONE" >> $LOG
