#!/bin/bash
# queue_5080_site.sh -- 2026-10-07 09:45 (Neuro 53, energy weakness). After the 5080 p26u / p26v queues: per-site
# threshold sensitivity (zero-shot, 7L p25o and 12L r26c), then p26w = conversion with head / MLP-input thresholds 2x.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python
while pgrep -f "bash temp/queue_5080_p26.sh" > /dev/null; do sleep 60; done
SRC=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
CUDA_VISIBLE_DEVICES=1 $PY temp/site_sens_eval.py $SRC configs/screen_p26k_7L_spk_kd_qv_sc20.yaml 2.0 2>&1 | grep -E 'base|site' | sed "s/^/  site-sens p25o /" >> $LOG
CUDA_VISIBLE_DEVICES=1 $PY temp/site_sens_eval.py models/checkpoints/r26c_13L_H6_chunk1024_150m/step_0012208.pt configs/r26d_13L_RC_conv_150m.yaml 2.0 2>&1 | grep -E 'base|site' | sed "s/^/  site-sens r26c /" >> $LOG
if [ "$(date +%H%M)" -ge 1700 ]; then echo "[n53 $(date '+%m-%d %H:%M:%S')] p26w SKIPPED (late)" >> $LOG; exit 0; fi
r=screen_p26w_7L_spk_kd_qv_site
SITE_SCALE="head=2,up=2" SCALE=1.0 bash temp/queue_spk_conv.sh 1 $r $SRC
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=1 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5080 site queue DONE" >> $LOG
