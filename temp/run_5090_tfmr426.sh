#!/bin/bash
# run_5090_tfmr426.sh -- 2026-09-28 (Glenn): start the 426M transformer baseline on the 5090 ONLY after every ppl-gap
# screen stream (run_5090_gap_stream.sh) has finished. Resumable: SIGINT the train_gpu process to pause (saves a ckpt);
# re-run this script to resume from the latest ckpt. WATCH: tail -f /home/glenn/projects/neuro/logs/baseline_tfmr426_L17.log
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
exec 9>logs/tfmr426.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[tfmr426 $(date +%H:%M:%S)] $*" | tee -a logs/gap_5090.log; }
say "waiting for all 5090 screen streams to finish"
while pgrep -f "[r]un_5090_gap_stream" > /dev/null; do sleep 60; done
say "screens done -> START baseline_tfmr426_L17 (resumes if ckpts exist)"
"$PY" src/train_gpu.py --config configs/baseline_tfmr426_L17.yaml --device cuda >> logs/baseline_tfmr426_L17.log 2>&1
say "baseline_tfmr426_L17 exit=$?"
