#!/bin/bash
# run_amb_sweep.sh -- 2026-09-27: isolate activation_memory_budget effect (B8 amb0.5 vs B8 baseline 3,415)
# and push batch (B20 amb0.3) on the 5080, v4 draft compiled.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
run(){ # tag batch amb
  $PY temp/profile_train.py --config configs/flagship_v4_363m.draft.yaml --tag $1 --batch $2 --wait 40 --active 20 --noprof --amb $3 > logs/prof_$1.log 2>&1
  rm -rf models/checkpoints/screen_prof_$1
  echo "$1: $(grep -h median logs/prof_$1.log) $(grep -c 'OutOfMemoryError: CUDA' logs/prof_$1.log) OOM | $(grep -E '^step' logs/prof_$1.log | tail -1 | grep -o 'vram.*')"
}
run v4d_b8_amb05 8 0.5
run v4d_b20_amb03 20 0.3
# repeat pair back-to-back for a fair B16-amb vs B12-plain comparison (run-to-run noise is large)
run v4d_b16_amb05_r2 16 0.5
run v4d_b12_r2 12 1.0
