#!/bin/bash
# queue_5080_energy.sh -- 2026-10-06 23:25 (Neuro 53). After the 5080 p26 queue: 7L conversions at higher thresholds
# (p26k scale 2.0, p26l scale 3.0), each followed by the honest energy eval (weight reads, temp/energy_eval_v2.py).
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
SRC=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
while pgrep -f "bash temp/queue_5080_p26.sh" > /dev/null; do sleep 60; done
for rs in "screen_p26k_7L_spk_kd_qv_sc20 2.0" "screen_p26l_7L_spk_kd_qv_sc30 3.0"; do
  set -- $rs
  [ "$(date +%H%M)" -ge 1800 ] && [ "$(date +%H%M)" -lt 2359 ] && break      # Glenn's 5080 evening
  SCALE=$2 bash temp/queue_spk_conv.sh 1 $1 $SRC
  ck=$(ls models/checkpoints/$1/step_*.pt | tail -1)
  CUDA_VISIBLE_DEVICES=1 $PY temp/energy_eval_v2.py "$1=$ck" "p25o=$SRC" 2>&1 | grep energy-v2 | sed "s/^/  /" >> logs/gap_queue.log
done
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5080 energy queue (p26k, p26l) DONE" >> logs/gap_queue.log
