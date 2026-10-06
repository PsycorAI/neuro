#!/bin/bash
# queue_5090_r25y.sh -- 2026-10-06 15:45 (Glenn: test the factorised head with training). After the KD reruns queue
# (r25w): r25y head 256, then head 512 (SCALE 1.5, KD, L8, q/v coded, 80M conv).
cd /home/glenn/projects/neuro
while pgrep -f "bash temp/queue_kd_reruns.sh" > /dev/null; do sleep 60; done
for r in 256 512; do
  SCALE=1.5 bash temp/queue_spk_conv.sh 0 r25y_13L_spk_kd_head${r}_150m models/checkpoints/r25o_13L_H6only_150m/step_0012208.pt
done
