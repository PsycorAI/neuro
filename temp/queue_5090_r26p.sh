#!/bin/bash
# queue_5090_r26p.sh -- 2026-10-09 (Neuro 54). After the outage queue (r26k -> r26m + evals) has exited: r26p = RC3
# dense (r26n) + head_weight_decay 1.0 on the 5090, then the same evals as r26n. ~3.7 h + evals.
# Launch: setsid -f bash temp/queue_5090_r26p.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
while pgrep -f "^bash temp/queue_5090_after_outage.sh" > /dev/null; do sleep 60; done
r=r26p_13L_RC3_headwd_150m
echo "[n54 $(date '+%m-%d %H:%M:%S')] 5090: $r (r26n + head_weight_decay 1.0; no-harm test for v4_cloud)" >> $LOG
GPU=0 FREE_MIN=1 UNTIL="2026-10-12 06:00" bash temp/run_5080_sched.sh $r
if [ -f models/checkpoints/$r/DONE ]; then
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  grep -E 'head_weight_decay|wnorm' logs/$r.log | sed -n '1p;$p' | sed "s/^/  [$r] /" >> $LOG
  echo "[n54 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
fi
