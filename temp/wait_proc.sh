#!/bin/bash
# wait_proc.sh -- 2026-10-08 (Neuro 53). Block until the trainer of run $1 has exited (or $2 minutes pass), then print
# its last step / eval lines and whether it finished ("done:") or crashed. The pgrep pattern starts with the python
# script path, which never appears in this waiter's own command line -> no self-match.
# Usage: bash temp/wait_proc.sh <run_name> [max_minutes]
cd /home/glenn/projects/neuro
r=$1; end=$(( $(date +%s) + ${2:-110} * 60 ))
while pgrep -f "src/train_gpu.py --config configs/$r.yaml" > /dev/null && [ "$(date +%s)" -lt "$end" ]; do sleep 60; done
date
if pgrep -f "src/train_gpu.py --config configs/$r.yaml" > /dev/null; then echo "STILL RUNNING after ${2:-110} min"; fi
grep '^step' logs/$r.log | tail -1
grep 'eval\] val_ppl' logs/$r.log | tail -2 | cut -c1-110
grep -q '^done:' logs/$r.log && echo "FINISHED (done:)" || { echo "NO done: line"; grep -E 'Error|Traceback' logs/$r.log | tail -2; }
exit 0
