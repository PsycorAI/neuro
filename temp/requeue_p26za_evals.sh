#!/bin/bash
# requeue_p26za_evals.sh -- 2026-10-10 (Neuro 54). One-shot: take screen_p26za_7L_RC3_x50 off the 5080 chain's done
# list, so the chain picks it again after p26zb: training is skipped (DONE marker), the evals run on a FREE 5080.
cd /home/glenn/projects/neuro
grep -vx screen_p26za_7L_RC3_x50 temp/queue_5080_n54.done > temp/queue_5080_n54.done.tmp; mv temp/queue_5080_n54.done.tmp temp/queue_5080_n54.done
echo "done list:"; cat temp/queue_5080_n54.done
pgrep -af "^bash temp/queue_5080_n54|src/train_gpu.py --config configs/screen_p26zb" | cut -c1-100
tail -n 2 logs/gap_queue.log | cut -c1-120
