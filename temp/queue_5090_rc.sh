#!/bin/bash
# queue_5090_rc.sh -- 2026-10-07 09:40 (Neuro 53; Glenn: RC, then strengthen ppl / energy / throughput). After r26t:
# r26d = RC conversion (r26c + r25x recipe) + honest energy, then r26e = ppl test (xchunk 0.25 from 50%).
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python
while pgrep -f "bash temp/queue_5090_r26t.sh" > /dev/null; do sleep 60; done
GPU=0 bash temp/bench_rc_tps.sh
SRC=models/checkpoints/r26c_13L_H6_chunk1024_150m/step_0012208.pt
r=r26d_13L_RC_conv_150m
SCALE=1.0 T=2048 bash temp/queue_spk_conv.sh 0 $r $SRC
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" "r26c=$SRC" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
for s in xcopy_sweep_eval multichunk_eval; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done
r=r26e_13L_H6_chunk1024_xlate_150m
GPU=0 bash temp/run_gap_queue.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
GPU=0 T=2048 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
CUDA_VISIBLE_DEVICES=0 $PY temp/multichunk_eval.py "$r=$ck" >> logs/e123_diag.log 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5090 RC queue (r26d, r26e) DONE" >> $LOG
