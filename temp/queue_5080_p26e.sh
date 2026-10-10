#!/bin/bash
# queue_5080_p26e.sh -- 2026-10-06 (Neuro 52; Glenn OK): p26e (scattered whole-vocab xchunk MQAR) on the 5080, then the
# r25 evals + diagnostics E1-E3 on p26e and its control p25o. Exits when done (Glenn takes the 5080 after this run).
cd /home/glenn/projects/neuro
r=screen_p26e_7L_xscatter
GPU=1 UNTIL="2026-10-07 06:00" bash temp/run_5080_sched.sh $r
ck=$(ls models/checkpoints/$r/step_*.pt 2>/dev/null | tail -1)
ctl=screen_p25o_7L_chunk512_x_H3add=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
if [ -f models/checkpoints/$r/DONE ]; then
  GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
  export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
  PY=/home/glenn/projects/bdh/venv/bin/python
  for s in xcopy_sweep_eval mem_write_probe multichunk_eval xcopy_random_inwin; do
    $PY temp/$s.py "p26e=$ck" "p25o=${ctl#*=}" >> logs/e123_diag.log 2>&1
  done
  $PY temp/xcopy_sweep_eval.py --contents none "p26e=$ck" "p25o=${ctl#*=}" >> logs/e123_diag.log 2>&1
fi
echo "[s1 $(date '+%m-%d %H:%M:%S')] p26e queue DONE (diagnostics in logs/e123_diag.log) -- 5080 free for Glenn" >> logs/gap_queue.log
