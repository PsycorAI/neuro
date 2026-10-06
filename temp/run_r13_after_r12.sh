#!/bin/bash
# run_r13_after_r12.sh -- 2026-09-30: start the r13 depth screens only once r12 holds the GPU-1 lock, so the
# (non-FIFO) flock cannot hand the GPU to r13 before r12. Then r13 waits for scheduler D2 (r12, r11a-c) to finish.
cd /home/glenn/projects/neuro
until grep -q '\[screen_r12_signed_canon_gapw015_s0_30m\] START' logs/gap_queue.log; do sleep 60; done
GPU=1 exec bash temp/run_5080_sched.sh screen_r13a_signed_3L_attn2_s0_30m screen_r13b_signed_3L_dsinit_s0_30m screen_r13d_signed_n4096_s0_30m
