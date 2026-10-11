#!/bin/bash
# queue_spk_conv.sh -- 2026-10-06 09:45 (energy plan, conversion). Usage: bash temp/queue_spk_conv.sh <GPU> <run> <src_ckpt> [wait_pattern]
# Seeds <run> from <src_ckpt> with calibrated spike codes (temp/mk_spike_seed.py), trains it (resume semantics), then
# recall / haystack / r25 / energy evals. Waits while a process matching wait_pattern runs.
cd /home/glenn/projects/neuro
G=$1; r=$2; SRC=$3; W=${4:-}
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python
[ -n "$W" ] && while pgrep -f "$W" > /dev/null; do sleep 60; done
rm -rf models/checkpoints/$r
CUDA_VISIBLE_DEVICES=$G $PY temp/mk_spike_seed.py $SRC models/checkpoints/$r configs/$r.yaml ${SCALE:-1.0} 2>&1 | grep OK | sed "s/^/  [$r] seed /" >> $LOG
echo "[g$G $(date '+%m-%d %H:%M:%S')] [$r] START" >> $LOG
CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 $PY src/train_gpu.py --config configs/$r.yaml --device cuda > logs/$r.log 2>&1
rc=$?   # 10-11: "exit=$?" inside the echo came after its $(date) and always printed 0
echo "[g$G $(date '+%m-%d %H:%M:%S')] [$r] exit=$rc :: $(grep 'eval\] val_ppl' logs/$r.log | awk '{print $3}' | tr '\n' ' ')" >> $LOG
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=$G $PY scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$r" | head -1 | sed "s/^/  recall /" >> $LOG
CUDA_VISIBLE_DEVICES=$G $PY temp/haystack_recall_eval.py $r=$ck 2>&1 | grep -E "^$r" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
GPU=$G bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
CUDA_VISIBLE_DEVICES=$G $PY temp/energy_eval.py "$r=$ck" "src=$SRC" 2>&1 | grep energy | sed "s/^/  energy /" >> $LOG
echo "[g$G $(date '+%m-%d %H:%M:%S')] $r queue DONE" >> $LOG
