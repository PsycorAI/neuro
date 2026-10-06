#!/bin/bash
# restart_5080_queue.sh -- 2026-09-30 (Glenn: 5080 chains must check the GPU is free before starting). Swaps the two
# OLD-logic schedulers (D2 = r12,r11a-c; r13 = r13a,b,d) for ONE new-logic scheduler with the same order, at a safe moment:
# after the 18:55 pause, when no screen is training (the schedulers are just sleeping until midnight). r12 resumes from
# its pause ckpt (no DONE marker); nothing is lost.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
OLD="181297 187225"
until [ $((10#$(date +%H%M))) -ge 1856 ] && ! pgrep -f "[s]rc/train_gpu.py --config configs/screen_" > /dev/null; do sleep 30; done
for p in $OLD; do pkill -P $p; kill $p; done      # children too: an orphaned sleep/flock would keep holding fd 9 (the lock)
sleep 3
echo "[s1 $(date '+%m-%d %H:%M:%S')] restart_5080_queue: replaced old schedulers ($OLD) with one free-GPU-checking scheduler" >> $LOG
GPU=1 exec bash temp/run_5080_sched.sh screen_r12_signed_canon_gapw015_s0_30m screen_r11a_hinge10_s0_30m \
  screen_r11b_ramp05_s0_30m screen_r11c_ramp10_s0_30m screen_r13a_signed_3L_attn2_s0_30m \
  screen_r13b_signed_3L_dsinit_s0_30m screen_r13d_signed_n4096_s0_30m
