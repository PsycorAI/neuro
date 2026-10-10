#!/bin/bash
# queue_5090_after_outage.sh -- 2026-10-09 15:50 (Neuro 53). Power outage ~10:25. 5090: resume r26o (step 16000) ->
# its evals + honest energy -> resume r26k -> r26m (12L spills on the 5080; Glenn has the 5080 until midnight).
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
post(){ r=$1; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  CUDA_VISIBLE_DEVICES=0 $PY scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$r" | head -1 | sed "s/^/  recall /" >> $LOG
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  CUDA_VISIBLE_DEVICES=0 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1; }
echo "[n53 $(date '+%m-%d %H:%M:%S')] power outage ~10:25 -> 5090 resumes r26o (step 16000) -> r26k -> r26m" >> $LOG
r=r26o_13L_RC3_conv_llamakd_150m
GPU=0 FREE_MIN=1 UNTIL="2026-10-11 06:00" bash temp/run_5080_sched.sh $r
if [ -f models/checkpoints/$r/DONE ]; then
  post $r; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] energy-v2 + diagnostics DONE" >> $LOG
fi
for r in r26k_13L_RC2_hola_seed1_150m r26m_13L_RC2_hola256_150m; do
  GPU=0 FREE_MIN=1 UNTIL="2026-10-11 06:00" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] && { post $r; echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG; }
done
