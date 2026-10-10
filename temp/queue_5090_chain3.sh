#!/bin/bash
# queue_5090_chain3.sh -- 2026-10-08 03:50 (Neuro 53). After r26f trains: r26f evals -> r26j RC2 conversion (HOLA) ->
# r26g RC conversion with Llama KD -> r26e xchunk dose-down.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python; export T=2048
diag(){ for s in "$@"; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done; }
evals(){ ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
         GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG; }
while pgrep -f "^bash temp/run_gap_queue.sh r26f_13L_H6_chunk1024_rand128_150m$" > /dev/null; do sleep 60; done
r=r26f_13L_H6_chunk1024_rand128_150m; evals; diag xcopy_sweep_eval mem_write_probe multichunk_eval
CUDA_VISIBLE_DEVICES=0 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
for rs in "r26j_13L_RC2_conv_150m r26h_13L_RC_hola64_150m" "r26g_13L_RC_conv_llamakd_150m r26c_13L_H6_chunk1024_150m"; do
  set -- $rs; r=$1
  SCALE=1.0 bash temp/queue_spk_conv.sh 0 $r models/checkpoints/$2/step_0012208.pt; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
  diag xcopy_sweep_eval multichunk_eval
done
r=r26e_13L_H6_chunk1024_xlate_150m
GPU=0 bash temp/run_gap_queue.sh $r; evals; diag multichunk_eval
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5090 chain3 DONE" >> $LOG
