#!/bin/bash
# queue_5080_p25mn.sh -- 2026-10-05 21:05 (Glenn: 5080 free). p25m (7L, H3 inserted) and p25n (H3 only) + r25 evals.
cd /home/glenn/projects/neuro
while pgrep -f "bash temp/queue_spk_conv.sh 1 " > /dev/null; do sleep 60; done
for r in screen_p25r_7L_H3add_T512c256; do
  GPU=1 UNTIL="2026-10-06 18:00" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=1 ITERS=20 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
done
echo "[s1 $(date '+%m-%d %H:%M:%S')] p25r queue DONE" >> logs/gap_queue.log
