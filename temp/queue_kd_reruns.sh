#!/bin/bash
cd /home/glenn/projects/neuro
( while pgrep -f "bash temp/queue_5080_p25uv.sh" > /dev/null; do sleep 60; done
  [ "$(date +%H%M)" -lt 1630 ] && SCALE=1.5 bash temp/queue_spk_conv.sh 1 screen_p25u_7L_spk_kd \
      models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt ) &
( while pgrep -f "bash temp/queue_5090_r25v.sh" > /dev/null; do sleep 60; done
  SCALE=1.5 bash temp/queue_spk_conv.sh 0 r25w_13L_spk_kd_sc15_150m models/checkpoints/r25o_13L_H6only_150m/step_0012208.pt ) &
wait
