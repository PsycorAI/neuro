#!/bin/bash
# run_sftw_5080.sh -- 2026-09-26, 5080: n=200 tool evals of the existing arms, then the v3-matched
# 426M transformer tok/s+VRAM probe (B8 then B4), then drain the shared tool-eval queue.
export CUDA_VISIBLE_DEVICES=1
source /home/glenn/projects/neuro/temp/sftw_lib.sh
say "5080 START"
eval_pass
for B in 8 4; do
  run=probe_tfmr426_b$B
  rm -rf models/checkpoints/$run
  say "$run START"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 -lms 2000 > logs/$run.smi &
  smi=$!
  $PY src/train_gpu.py --config temp/$run.yaml --device cuda > logs/$run.log 2>&1
  rc=$?
  kill $smi
  say "$run exit=$rc peak_sys_MiB=$(sort -n logs/$run.smi | tail -n 1) :: $(grep -E '^step' logs/$run.log | tail -n 1)"
  rm -rf models/checkpoints/$run
done
eval_until_done
say "5080 DONE"
