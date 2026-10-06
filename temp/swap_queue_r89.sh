#!/bin/bash
# swap_queue_r89.sh -- 2026-09-29 (Glenn OK): at tonight's 18:55 pause, replace the 5080 queue with the signed-first order.
# Waits for gap128's pause line, checks no trainer is running on GPU 1, stops the old scheduler AND its children (a
# sleeping child inherits the flock fd and would hold the GPU lock until midnight), then execs the new scheduler.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
until grep -q 'screen_r7_gap128_s0_30m\] paused at' $LOG; do sleep 30; done
sleep 10
if pgrep -f 'train_gpu.py --config configs/screen_' >/dev/null; then echo "[swap] ABORT: a screen trainer is still running" | tee -a $LOG; exit 1; fi
old=$(pgrep -f '^bash /home/glenn/projects/neuro/temp/run_5080_sched.sh' | head -1)
[ -n "$old" ] || { echo "[swap] ABORT: old scheduler not found" | tee -a $LOG; exit 1; }
pkill -P "$old"; kill "$old"; sleep 3
echo "[swap $(date '+%m-%d %H:%M:%S')] old scheduler $old stopped; launching signed-first r8/r9 queue (Glenn OK)" | tee -a $LOG
exec env GPU=1 bash /home/glenn/projects/neuro/temp/run_5080_sched.sh screen_r7_gap128_s0_30m screen_r8_signed_s1_30m \
  screen_r8_signed_lam2_s0_30m screen_r9a_stateful_s0_30m screen_r9b_swastoch_s0_30m screen_r9c_stateful_swa_s0_30m \
  screen_r8_signed_3L_s0_30m screen_r8_signed_canon_s0_30m screen_r7_gapw015_s0_30m
