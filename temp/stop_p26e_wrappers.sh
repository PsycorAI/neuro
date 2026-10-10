#!/bin/bash
# stop_p26e_wrappers.sh -- 2026-10-06 (Glenn needs the 5080 right after p26e training; evals deferred).
# SIGTERM the queue wrapper + scheduler ONLY (exact pids, cmdline-checked); train_gpu keeps running.
cd /home/glenn/projects/neuro
for p in $(pgrep -f '^bash temp/queue_5080_p26e.sh$') $(pgrep -f '^bash temp/run_5080_sched.sh screen_p26e_7L_xscatter$'); do
  echo "SIGTERM $p: $(tr '\0' ' ' < /proc/$p/cmdline)"; kill -TERM $p
done
sleep 2
pgrep -af 'queue_5080_p26e|run_5080_sched' || echo "wrappers gone"
pgrep -af 'train_gpu.py --config configs/screen_p26e' || echo "WARNING: training not running"
