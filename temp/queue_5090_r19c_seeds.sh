#!/bin/bash
# queue_5090_r19c_seeds.sh -- 2026-10-03: r19c seed repeats on the 5090 AFTER queue_5090_r19speed.sh (waits for its
# 'r19speed DONE' line, so the order is deterministic), then run_gap_queue.sh (takes the gpu0 lock with flock -n).
cd /home/glenn/projects/neuro
until grep -q 'r19speed DONE' logs/night_5090.log; do sleep 60; done
sleep 30
GPU=0 bash temp/run_gap_queue.sh screen_r19c_layer10_s1_30m screen_r19c_layer10_s2_30m
