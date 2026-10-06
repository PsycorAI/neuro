#!/bin/bash
# queue_5090_r21b.sh -- 2026-10-04: second half of round 21 on the 5090 (the 5080 cannot fit the gap-MQAR / candidate
# arms at the G3 shape). Waits for the first r21 queue (run_gap_queue.sh r21a r21b r21c, PID given as $1) to exit.
cd /home/glenn/projects/neuro
while kill -0 "$1" 2>/dev/null; do sleep 60; done
sleep 10
GPU=0 bash temp/run_gap_queue.sh r21e_tfmr_w128x1_150m r21d_tfmr_x1_150m r21f_w128_150m
