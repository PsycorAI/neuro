#!/bin/bash
# queue_5090_chain.sh -- 2026-10-07 15:45 (Neuro 53). One 5090 chain (HOLA first, Glenn): r26d RC conversion -> r26h RC +
# HOLA -> r26f RC + stochastic chunk -> r26e xchunk dose-down -> r26g RC conversion with Llama-3.2-1B KD.
cd /home/glenn/projects/neuro
LOG=logs/gap_queue.log; PY=/home/glenn/projects/bdh/venv/bin/python; export T=2048
SRC=models/checkpoints/r26c_13L_H6_chunk1024_150m/step_0012208.pt
diag(){ for s in "$@"; do CUDA_VISIBLE_DEVICES=0 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1; done; }
evals(){ ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
         GPU=0 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG; }
while pgrep -f "bash temp/bench_rc_tps.sh" > /dev/null; do sleep 30; done
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5090 chain start: r26d -> r26h -> r26f -> r26e -> r26g" >> $LOG
r=r26d_13L_RC_conv_150m
SCALE=1.0 bash temp/queue_spk_conv.sh 0 $r $SRC; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
diag xcopy_sweep_eval multichunk_eval
for r in r26h_13L_RC_hola64_150m r26f_13L_H6_chunk1024_rand128_150m r26e_13L_H6_chunk1024_xlate_150m; do
  GPU=0 bash temp/run_gap_queue.sh $r; evals; diag xcopy_sweep_eval mem_write_probe multichunk_eval
  CUDA_VISIBLE_DEVICES=0 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
  echo "[n53 $(date '+%m-%d %H:%M:%S')] [$r] evals + diagnostics DONE" >> $LOG
done
r=r26g_13L_RC_conv_llamakd_150m
SCALE=1.0 bash temp/queue_spk_conv.sh 0 $r $SRC; ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
CUDA_VISIBLE_DEVICES=0 $PY temp/energy_eval_v2.py "$r=$ck" 2>&1 | grep energy-v2 | sed "s/^/  /" >> $LOG
diag xcopy_sweep_eval multichunk_eval
echo "[n53 $(date '+%m-%d %H:%M:%S')] 5090 chain DONE" >> $LOG
