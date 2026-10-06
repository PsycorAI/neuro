#!/bin/bash
# run_1b_probe_expseg.sh -- 2026-09-26: 1B A' at B=12 with expandable_segments. Plain B=12
# reserved 33.7GB for 12.7GB allocated (spill, 2.1k tok/s); B=8 = 4.0k tok/s @28.2GB; B=4 =
# 2.2k -> overhead-bound, so fitting B=12 under the cliff is the cheapest v4 speed lever.
# Waits for the 5090 SFT screens (run_sftab2.sh) to finish first.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/probe_1b.log
say(){ echo "[probe1b $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
while pgrep -f '[r]un_sftab' > /dev/null; do sleep 30; done
run=probe_1b_b12_expseg
sed -e 's/^run_name: .*/run_name: probe_1b_b12_expseg/' temp/probe_1b_b12.yaml > temp/$run.yaml
rm -rf models/checkpoints/$run
say "$run START (PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 2000 > logs/$run.smi &
smi=$!
$PY src/train_gpu.py --config temp/$run.yaml --device cuda > logs/$run.log 2>&1
rc=$?
kill $smi
say "$run exit=$rc peak_sys_MiB=$(sort -n logs/$run.smi | tail -n 1)"
grep -E 'tok/s|expandable|out of memory' logs/$run.log | tail -n 4 | tee -a "$LOG"
rm -rf models/checkpoints/$run
say "expseg DONE"
