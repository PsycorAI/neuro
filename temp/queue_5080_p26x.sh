#!/bin/bash
# queue_5080_p26x.sh -- 2026-10-07 11:30 (Neuro 53, energy). After the 5080 site queue: p26x = p26u with frozen thresholds
# (2x, 120M tok) + honest energy. Must start by 15:45 to finish before Glenn's evening.
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
SRC=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
r=screen_p26x_7L_spk_kd_qv_sc20_120M_frz
while pgrep -f "bash temp/queue_5080_(site|p26).sh" > /dev/null; do sleep 60; done
if [ "$(date +%H%M)" -ge 1545 ]; then echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] SKIPPED (too late)" >> logs/gap_queue.log; exit 0; fi
SCALE=2.0 bash temp/queue_spk_conv.sh 1 $r $SRC
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=1 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> logs/gap_queue.log
echo "[n53 $(date '+%m-%d %H:%M:%S')] p26x queue DONE -- 5080 free" >> logs/gap_queue.log
