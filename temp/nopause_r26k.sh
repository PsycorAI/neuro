#!/bin/bash
# nopause_r26k.sh -- 2026-10-08 13:10 (Glenn: 5080 free all night). Drop r26k's 17:50 pause: stop its queue wrapper and
# scheduler (the trainer keeps running), then a plain waiter runs the evals when it finishes.
cd /home/glenn/projects/neuro
for pat in "^bash temp/queue_5080_r26k.sh$" "^bash temp/run_5080_sched.sh r26k_13L_RC2_hola_seed1_150m$"; do
  for p in $(pgrep -f "$pat"); do echo "TERM $p"; kill -TERM $p; done; done
cat > temp/after_r26k.sh << 'EOS'
#!/bin/bash
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
r=r26k_13L_RC2_hola_seed1_150m
while pgrep -f "configs/$r.yaml" > /dev/null; do sleep 60; done
echo "[s1 $(date '+%m-%d %H:%M:%S')] [$r] exit :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
grep -q '^done:' logs/$r.log || { echo "[n53] [$r] ENDED WITHOUT done:" >> $LOG; exit 1; }
touch models/checkpoints/$r/DONE; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=1 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
CUDA_VISIBLE_DEVICES=1 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
EOS
setsid -f bash temp/after_r26k.sh > /dev/null 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26k: evening pause removed (Glenn: 5080 free all night); evals via temp/after_r26k.sh" >> logs/gap_queue.log
sleep 1; ps -eo pid,args | grep -E "r26k|after_r26k" | grep -v grep | grep -v nopause | cut -c1-110
