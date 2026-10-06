#!/bin/bash
# queue_5090_r25l.sh -- 2026-10-05 17:25. After queue_5090_r25k.sh: r25l (late insertion of a memory block at depth 6).
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
PY=/home/glenn/projects/bdh/venv/bin/python
while pgrep -f "bash temp/queue_5090_r25k.sh" > /dev/null; do sleep 60; done
r=r25l_late_insH6_150m
rm -rf models/checkpoints/$r
$PY temp/mk_insert_ckpt.py models/checkpoints/r21b_x1_150m/step_0012208.pt models/checkpoints/$r configs/$r.yaml >> $LOG 2>&1
echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] START (r21b + inserted H6)" >> $LOG
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY src/train_gpu.py --config configs/$r.yaml --device cuda > logs/$r.log 2>&1
echo "[q0 $(date '+%m-%d %H:%M:%S')] [$r] exit=$? :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$r" | head -1 \
    | sed "s/^/  recall /" >> $LOG
CUDA_VISIBLE_DEVICES=0 $PY temp/haystack_recall_eval.py $r=$ck 2>&1 | grep -E "^$r" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25l queue DONE" >> $LOG
