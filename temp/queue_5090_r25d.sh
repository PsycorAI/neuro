#!/bin/bash
# queue_5090_r25d.sh -- 2026-10-05 08:00: r25d (12L chunk + xchunk from 0 + late read [4,8]) + evals; GPU-0 lock.
cd /home/glenn/projects/neuro
flock logs/gap_queue_gpu0.lock true
for r in r25d_chunk_x1e_read_150m; do
  GPU=0 bash temp/run_gap_queue.sh $r
  ck=$(ls models/checkpoints/$r/step_*.pt 2>/dev/null | tail -1)
  [ -n "$ck" ] && GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
done
echo "[q0 $(date '+%m-%d %H:%M:%S')] r25d queue DONE" >> logs/gap_queue.log
