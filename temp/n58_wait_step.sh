#!/bin/bash
# n58_wait_step.sh <run> <step> [max_min=9] -- Neuro 58: block until the run's log has a step line >= <step> written
# AFTER this script started (so old lines of an earlier attempt do not count), or an error, or the trainer is gone.
cd /home/glenn/projects/neuro || exit 1
L=logs/$1.log; n0=$(wc -l < $L); end=$(( $(date +%s) + ${3:-9} * 60 ))
hit(){ tail -n +$((n0 + 1)) $L | awk -v s=$2 '/^step [0-9]+ \|/ && $2 >= s {f=1} END{exit !f}'; }
until hit "$@" || tail -n +$((n0 + 1)) $L | grep -a -q -E 'Traceback (most recent call last):$|^[A-Za-z]*Error:' || ! pgrep -f "src/train_gpu.py --config configs/$1.yaml" > /dev/null || [ "$(date +%s)" -ge "$end" ]; do sleep 15; done
date; grep -a -E '^resumed|remapped|warn' $L | tail -n 2 | cut -c1-130
grep -a -E '^step|eval\]|\[wnorm\]|\[mlp\]|Error|Traceback' $L | sed -E 's/ thr=[0-9.]+//g; s/rate=//g' | tail -n ${4:-4} | cut -c1-200
free -g | sed -n 2p; echo "home scripts alive: $(pgrep -f '^bash (temp|cloud)/' | wc -l)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
pgrep -f "src/train_gpu.py --config configs/$1.yaml" > /dev/null || echo "TRAINER NOT RUNNING"
