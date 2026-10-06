#!/bin/bash
# run_heal_v3.sh -- 2026-09-27: heal v3 after the bf16 future-leak fix (commit 2e98665).
# Seeds models/checkpoints/heal_v3_363m with a HARDLINK of the 10B v3 ckpt (no copy, original untouched;
# keep_last pruning only removes the link), then train_gpu resumes it: step 271268 -> +1B tok (~28h @ 9.7k tok/s).
# Model + optimizer state + head-norm/val_ppl watchdogs as in v3; watchdog baseline starts at the first HONEST eval.
# WATCH: tail -f /home/glenn/projects/neuro/logs/heal_v3_363m.log
set -eu
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
CK=models/checkpoints; SRC=$CK/flagship_v3_363m/step_0271268.pt; DST=$CK/heal_v3_363m
git merge-base --is-ancestor 2e98665 HEAD || { echo "leak fix 2e98665 not in HEAD -- abort"; exit 1; }
mkdir -p $DST
[ -e $DST/step_0271268.pt ] || ls $DST/step_*.pt >/dev/null 2>&1 || ln $SRC $DST/step_0271268.pt
exec /home/glenn/projects/bdh/venv/bin/python src/train_gpu.py --config configs/heal_v3_363m.yaml --device cuda >> logs/heal_v3_363m.log 2>&1
