#!/bin/bash
# run_1b_probe.sh -- 2026-09-26: v4 A' 1B tok/s + VRAM probe on the 5090 (GPU 0).
# Widest micro-batch first; a failed/OOM arm is logged and the sweep continues.
# Checkpoints are deleted before/after (a leftover probe ckpt would auto-resume).
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/probe_1b.log
say(){ echo "[probe1b $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "START (GPU0 = 5090)"
for B in 12 8 4; do
  run=probe_1b_b$B
  rm -rf models/checkpoints/$run
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 2000 > logs/$run.smi &
  smi=$!
  CUDA_VISIBLE_DEVICES=0 $PY src/train_gpu.py --config temp/$run.yaml --device cuda > logs/$run.log 2>&1
  rc=$?
  kill $smi
  peak=$(sort -n logs/$run.smi | tail -n 1)
  say "$run exit=$rc peak_sys_MiB=$peak"
  grep -E 'params=|tok/s|out of memory|Error' logs/$run.log | tail -n 4 | tee -a "$LOG"
  rm -rf models/checkpoints/$run
done
say "DONE"
