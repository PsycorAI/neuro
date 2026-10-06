#!/bin/bash
cd /home/glenn/projects/neuro
while pgrep -f "run_5080_sched.sh screen_p25r" > /dev/null; do sleep 60; done
r=screen_p25r_7L_H3add_T512c256; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
[ -f models/checkpoints/$r/DONE ] && GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
bash temp/queue_spk_conv.sh 1 screen_p25s_7L_spk_conv models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
