#!/bin/bash
cd /home/glenn/projects/neuro
while pgrep -f "bash temp/queue_spk_conv.sh 1 " > /dev/null; do sleep 60; done
SCALE=1.5 bash temp/queue_spk_conv.sh 1 screen_p25u_7L_spk_kd models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
r=screen_p25v_7L_xcopy
GPU=1 UNTIL="2026-10-06 18:00" bash temp/run_5080_sched.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt 2>/dev/null | tail -1)
[ -f models/checkpoints/$r/DONE ] && GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
echo "[s1 $(date '+%m-%d %H:%M:%S')] p25u/v queue DONE" >> logs/gap_queue.log
