#!/bin/bash
# queue_5090_r26g.sh -- 2026-10-07 13:45 (Neuro 53, ppl). After queue_sc.sh (r26f): r26g = RC conversion with a Llama-3.2-1B
# teacher (vs r26d self-KD) + honest energy + diagnostics.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python
while pgrep -f "bash temp/queue_sc.sh" > /dev/null; do sleep 60; done
SRC=models/checkpoints/r26c_13L_H6_chunk1024_150m/step_0012208.pt
r=r26g_13L_RC_conv_llamakd_150m
SCALE=1.0 T=2048 bash temp/queue_spk_conv.sh 0 $r $SRC
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
echo "[n53 $(date '+%m-%d %H:%M:%S')] r26g queue DONE" >> $LOG
