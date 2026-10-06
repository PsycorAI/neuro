#!/bin/bash
# queue_5090_night3.sh -- 2026-10-03 02:05: GPU tail-fill after night2 (Glenn: "utilize the 5090 and 5080 tonight").
# Waits (blocking) for night2's logs/5090_seq.lock, then runs seed repeats on the 5090 (configs: temp/mk_r17_seeds_s2.py).
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_queue.log
cd /home/glenn/projects/neuro
exec 8>logs/5090_seq.lock
flock 8
echo "[nt $(date '+%m-%d %H:%M:%S')] night3: lock held" >> logs/night_5090.log
GPU=0 bash temp/run_gap_queue.sh screen_r16_l3_gelu_s1_30m screen_r17f_k688_s2_30m screen_r16_c0_s2_30m
echo "[nt $(date '+%m-%d %H:%M:%S')] NIGHT3 DONE" >> logs/night_5090.log
