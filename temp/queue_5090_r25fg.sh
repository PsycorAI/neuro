#!/bin/bash
# queue_5090_r25fg.sh -- 2026-10-05 10:25. After queue_5090_r25d.sh exits: late-memory-phase runs r25f (late read) and
# r25g (no late read), each seeded with r21b's final ckpt (the trainer resumes from the run dir's latest step_*.pt).
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
while pgrep -f "bash temp/queue_5090_r25d.sh" > /dev/null; do sleep 120; done
for r in r25f_late_chunk_read_150m r25g_late_chunk_150m; do
  mkdir -p models/checkpoints/$r
  cp models/checkpoints/r21b_x1_150m/step_0012208.pt models/checkpoints/$r/
  echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] START (late phase from r21b step 12208)" >> $LOG
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/glenn/projects/bdh/venv/bin/python src/train_gpu.py \
      --config configs/$r.yaml --device cuda > logs/$r.log 2>&1
  echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] exit=$? :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  CUDA_VISIBLE_DEVICES=0 /home/glenn/projects/bdh/venv/bin/python scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 \
      | grep -E "^ *$r" | head -1 | sed "s/^/  recall /" >> $LOG
  CUDA_VISIBLE_DEVICES=0 /home/glenn/projects/bdh/venv/bin/python temp/haystack_recall_eval.py $r=$ck 2>&1 | grep -E "^$r" \
      | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
  GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
done
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25f/g queue DONE" >> $LOG
