#!/bin/bash
# CPU plumbing smoke of the scorecard's autointerp step (1 unit per layer, Llama-3.2-3B-Instruct on CPU, low priority).
cd /home/glenn/projects/neuro
CUDA_VISIBLE_DEVICES= setsid -f nice -n 19 /home/glenn/projects/bdh/venv/bin/python scripts/interp_scorecard.py \
  --device cpu --threads 4 --B 4 --autointerp 1 r15c_ai=models/checkpoints/screen_r15c_topk16_s0_30m/step_0006104.pt \
  > logs/scorecard_autointerp_smoke.log 2>&1 < /dev/null
sleep 2; ps -eo pid,args | grep "[i]nterp_scorecard" | cut -c1-120
