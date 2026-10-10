#!/bin/bash
# queue_1009.sh -- 2026-10-09 07:45 (Neuro 53). 5090: r26o (RC3 + Llama KD conversion). 5080: resume r26k, then r26m,
# with max_split_size_mb (fragmentation drove reserved VRAM to 16.2 GB -> WDDM spill -> watchdog kills).
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
post(){ g=$1; r=$2; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=$g bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=$g $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  CUDA_VISIBLE_DEVICES=$g $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG; }
(
  r=r26o_13L_RC3_conv_llamakd_150m
  SCALE=1.0 bash temp/queue_spk_conv.sh 0 $r models/checkpoints/r26n_13L_RC2_xlate_150m/step_0012208.pt
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] energy-v2 + diagnostics DONE" >> $LOG
) &
(
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
  for r in r26k_13L_RC2_hola_seed1_150m r26m_13L_RC2_hola256_150m; do
    GPU=1 FREE_MIN=1 UNTIL="2026-10-10 06:00" bash temp/run_5080_sched.sh $r
    [ -f models/checkpoints/$r/DONE ] && post 1 $r
  done
) &
wait
