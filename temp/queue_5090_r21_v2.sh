#!/bin/bash
# queue_5090_r21_v2.sh -- 2026-10-04 15:10: round 21 after the Inductor fix (mqar_compile_static + gdn2_cumsum_eager_max_b 8;
# r21c crashed at step ~8000 when 512-token gap batches began). (1) RESUME r21c from step_8000 (train_gpu auto-resumes;
# the new flags are exact math), then its evals; (2) the rest via run_gap_queue (fresh starts), memory-forcing arms first.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_queue.log
say(){ echo "[q0 $(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }
run=r21c_w128x1_150m
say "[$run] RESUME from $(ls models/checkpoints/$run/step_*.pt | tail -1)"
"$PY" src/train_gpu.py --config configs/$run.yaml --device cuda >> logs/$run.log 2>&1
say "[$run] exit=$? :: $(grep 'eval\] val_ppl' logs/$run.log | awk '{print $3}' | tr '\n' ' ')"
ck=$(ls models/checkpoints/$run/step_*.pt 2>/dev/null | tail -1)
"$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 | sed "s/^/  recall /" >> $LOG
[ -n "$ck" ] && "$PY" temp/haystack_recall_eval.py $run=$ck 2>&1 | grep -E "^$run" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=0 bash temp/run_gap_queue.sh r21g_w32x1early_150m r21b_x1_150m r21i_x1early_150m r21e_tfmr_w128x1_150m \
    r21d_tfmr_x1_150m r21f_w128_150m
