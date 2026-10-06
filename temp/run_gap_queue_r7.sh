#!/bin/bash
# run_gap_queue_r7.sh -- 2026-09-29. Waits for the running GPU-1 queue (pid 121725, ends after screen_r5_signed ~16:50),
# then runs the round-7 queue. DEADLINE=2400 on purpose: the HHMM cutoff in run_gap_queue.sh can't cross midnight.
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_queue.log
while kill -0 121725 2>/dev/null; do sleep 60; done
GPU=1 DEADLINE=2400 bash /home/glenn/projects/neuro/temp/run_gap_queue.sh \
  screen_r7_gap128_s0_30m screen_r7_gapw015_s0_30m screen_r6_canon_s1_30m screen_r5_alpha1_s0_30m screen_r5_alpha4_s0_30m
