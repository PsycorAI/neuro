#!/bin/bash
# eval_p26e_deferred.sh -- 2026-10-06. p26e evals, run AFTER Glenn is done with the 5080 (GPU=1) or on another GPU.
# recall + haystack (as run_5080_sched would), r25 evals, E1-E3 + random-span control on p26e and the control p25o.
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
r=screen_p26e_7L_xscatter; LOG=logs/gap_queue.log
ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
ctl=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
if [ -z "${SKIP_RH:-}" ]; then   # SKIP_RH=1: run_5080_sched.sh already ran recall + haystack
  $PY scripts/recall_eval.py --runs $r --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$r" | head -1 | sed "s/^/  recall /" >> $LOG
  $PY temp/haystack_recall_eval.py $r=$ck 2>&1 | grep -E "^$r" | sed "s/^/  haystack (gap 0\/64\/192\/440) /" >> $LOG
fi
GPU=${GPU:-1} bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> $LOG
for s in xcopy_sweep_eval mem_write_probe multichunk_eval xcopy_random_inwin; do
  $PY temp/$s.py "p26e=$ck" "p25o=$ctl" >> logs/e123_diag.log 2>&1
done
$PY temp/xcopy_sweep_eval.py --contents none "p26e=$ck" "p25o=$ctl" >> logs/e123_diag.log 2>&1
echo "[s1 $(date '+%m-%d %H:%M:%S')] p26e deferred evals DONE (diagnostics in logs/e123_diag.log)" >> $LOG
