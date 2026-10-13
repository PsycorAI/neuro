#!/bin/bash
# Neuro 58: block (max $1 min, default 9) until the probe log has $2 (default 2) new-run step lines or an error; print state.
cd /home/glenn/projects/neuro || exit 1
L=logs/${3:-probe_v4_xchunk}.log; end=$(( $(date +%s) + ${1:-9} * 60 ))
cnt(){ awk '/^resumed /{n=0} /^step [0-9]+ \|/{n++} END{print n+0}' $L; }
until [ "$(cnt)" -ge "${2:-2}" ] || grep -a -q -E 'Traceback|Error' $L || ! pgrep -f 'src/train_gpu.py --config configs/probe_' > /dev/null || [ "$(date +%s)" -ge "$end" ]; do sleep 15; done
date; grep -a -E '^step|eval\]|\[beta\]|\[spike\]|Error|Traceback|saved' $L | tail -n 8 | cut -c1-230
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
pgrep -af 'src/train_gpu.py --config configs/probe_' | cut -c1-90 || echo "TRAINER NOT RUNNING"
