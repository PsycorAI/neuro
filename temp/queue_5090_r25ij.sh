#!/bin/bash
# queue_5090_r25ij.sh -- 2026-10-05 13:00: r25i (balanced late phase from r21b ckpt) then r25j (12L H0+H6 chunked, from
# scratch), each with recall / haystack / r25 evals. Watchdog-compatible (plain train_gpu processes).
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
PY=/home/glenn/projects/bdh/venv/bin/python
while pgrep -f "bash temp/queue_5090_r25h.sh" > /dev/null; do sleep 60; done
for r in r25i_late_read_xbal_150m r25j_chunk_H6_150m; do
  case $r in r25i*) mkdir -p models/checkpoints/$r && cp models/checkpoints/r21b_x1_150m/step_0012208.pt models/checkpoints/$r/;;
             *) rm -rf models/checkpoints/$r;; esac
  echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] START" >> $LOG
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY src/train_gpu.py --config configs/$r.yaml --device cuda > logs/$r.log 2>&1
  echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] exit=$? :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  CUDA_VISIBLE_DEVICES=0 $PY scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$r" | head -1 \
      | sed "s/^/  recall /" >> $LOG
  CUDA_VISIBLE_DEVICES=0 $PY temp/haystack_recall_eval.py $r=$ck 2>&1 | grep -E "^$r" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
done
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25i/j queue DONE" >> $LOG
