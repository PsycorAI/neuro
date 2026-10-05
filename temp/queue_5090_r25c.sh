#!/bin/bash
# queue_5090_r25c.sh -- 2026-10-05 05:35. After queue_5090_r25.sh (r25a, r25b) exits: r25c (r25a + NL-fact aux; the
# 6L NL run p25f spilled on the 5080) + its r25 evals.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log
while pgrep -f "bash temp/queue_5090_r25.sh" > /dev/null; do sleep 120; done
r=r25c_chunk_x1_nl_150m
GPU=0 bash temp/run_gap_queue.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt 2>/dev/null | tail -1)
[ -n "$ck" ] && GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> "$LOG"
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25c queue DONE" >> "$LOG"
