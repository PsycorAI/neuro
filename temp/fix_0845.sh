#!/bin/bash
# fix_0845.sh -- 12L runs keep spilling on the 5080 (r26k killed 3x). Stop the 5080 branch; resume r26k + r26m on the
# 5090 after r26o.
cd /home/glenn/projects/neuro
for p in $(pgrep -f "^bash temp/run_5080_sched.sh r26m_13L_RC2_hola256_150m$"); do echo "TERM sched $p"; kill -TERM $p; done
for p in $(pgrep -f "src/train_gpu.py --config configs/r26m_13L_RC2_hola256_150m.yaml"); do echo "INT train $p"; kill -INT $p; done
# the queue_1009.sh 5080 subshell: its child is (or was) a run_5080_sched; kill subshells that are not the r26o branch
for p in $(pgrep -f "^bash temp/queue_1009.sh$"); do
  if pstree -p $p 2>/dev/null | grep -q "queue_spk_conv"; then echo "keep r26o branch $p"; else
    kids=$(pgrep -P $p | wc -l); [ "$kids" -le 1 ] && pgrep -P $p > /dev/null && { echo "TERM 5080 branch $p"; kill -TERM $p; }; fi
done
sleep 5
cat > temp/queue_5090_resume_km.sh << 'EOS'
#!/bin/bash
cd /home/glenn/projects/neuro
export T=2048 PYTHONUNBUFFERED=1; PY=/home/glenn/projects/bdh/venv/bin/python; LOG=logs/gap_queue.log
until grep -q 'r26o_13L_RC3_conv_llamakd_150m\] energy-v2 + diagnostics DONE' $LOG; do sleep 120; done
for r in r26k_13L_RC2_hola_seed1_150m r26m_13L_RC2_hola256_150m; do
  GPU=0 FREE_MIN=1 UNTIL="2026-10-10 18:00" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
  for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
done
EOS
setsid -f bash temp/queue_5090_resume_km.sh > /dev/null 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26k killed 3rd time on the 5080 (2.4k tok/s): 12L unreliable on the 5080 -> r26k + r26m resume on the 5090 after r26o." >> logs/gap_queue.log
ps -eo pid,args | grep -E "queue_1009|queue_5090_resume|sched|src/train_gpu.py" | grep -v grep | cut -c1-110
