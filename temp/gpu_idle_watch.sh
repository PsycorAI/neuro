#!/bin/bash
# temp/gpu_idle_watch.sh -- 2026-10-28 (Glenn: "know when the gpu is idle, and alert you"): exits as soon as any GPU has
# been idle (utilization < 5% on every 30 s sample) for MIN minutes, printing the GPUs and running jobs, so the Claude
# session waiting on it is notified and can queue work. A job that holds a GPU but runs on the CPU counts as idle.
# Usage: bash temp/gpu_idle_watch.sh [MIN=10] [GPUS="0 1"]   (0 = 5090, 1 = 5080, nvidia-smi PCI order)
MIN=${1:-10}; GPUS=${2:-"0 1"}
declare -A idle
while true; do
  for g in $GPUS; do
    u=$(nvidia-smi -i "$g" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ "${u:-0}" -lt 5 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge $(( MIN * 2 )) ]; then
      echo "$(date '+%m-%d %H:%M') GPU $g idle for ${MIN}+ min"
      nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
      ps -eo pid,etime,cmd | grep -E 'python|bash temp' | grep -v grep | cut -c1-140
      exit 0
    fi
  done
  sleep 30
done
