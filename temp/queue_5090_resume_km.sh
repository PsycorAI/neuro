#!/bin/bash
# queue_5090_resume_km.sh -- 2026-10-09 08:50 (Neuro 53). After r26o's own conversion queue finishes: r26o honest energy
# + diagnostics, then resume r26k and r26m on the 5090 (12L spills on the 5080).
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
until grep -q 'r26o_13L_RC3_conv_llamakd_150m queue DONE' $LOG; do sleep 120; done
r=r26o_13L_RC3_conv_llamakd_150m; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] energy-v2 + diagnostics DONE" >> $LOG
for r in r26k_13L_RC2_hola_seed1_150m r26m_13L_RC2_hola256_150m; do
  GPU=0 FREE_MIN=1 UNTIL="2026-10-10 18:00" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
done
