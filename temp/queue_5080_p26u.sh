#!/bin/bash
# queue_5080_p26u.sh -- 2026-10-07 07:25 (Neuro 53). After the 5080 polish queue: p26u = 2x-threshold conversion for
# 120M tok (vs p26k 40M), then the honest energy eval. Skips if it cannot finish before Glenn's evening (start by 14:30).
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
SRC=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
r=screen_p26u_7L_spk_kd_qv_sc20_120M
while pgrep -f "bash temp/queue_5080_p26.sh" > /dev/null; do sleep 60; done
if [ "$(date +%H%M)" -ge 1430 ]; then echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] SKIPPED (too late to finish before 18:30)" >> logs/gap_queue.log; exit 0; fi
SCALE=2.0 bash temp/queue_spk_conv.sh 1 $r $SRC
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=1 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> logs/gap_queue.log
echo "[n53 $(date '+%m-%d %H:%M:%S')] p26u queue DONE -- 5080 free" >> logs/gap_queue.log
