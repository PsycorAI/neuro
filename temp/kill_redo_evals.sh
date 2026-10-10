#!/bin/bash
# kill_redo_evals.sh -- 2026-10-10 (Neuro 54). One-shot: stop the stuck p26za eval redo (it held the 5080 at 100% /
# 15 GB for 45 min and starved screen_p26zb). Kills only the redo script, its eval wrapper and its eval python.
pkill -TERM -f "^bash temp/redo_evals_p26za.sh"
pkill -TERM -f "^bash temp/run_r25_evals.sh screen_p26za_7L_RC3_x50="
pkill -TERM -f "temp/[a-z_]*eval.py screen_p26za_7L_RC3_x50="
sleep 5
pgrep -af "screen_p26za_7L_RC3_x50" | cut -c1-120
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
