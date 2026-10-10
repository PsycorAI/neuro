#!/bin/bash
# queue_5080_p26.sh -- 2026-10-06 (Neuro 53; Glenn: keep both GPUs busy overnight). Usage: bash temp/queue_5080_p26.sh <run> [...]
# Each 7L p26 run on the 5080 (pauses for Glenn at UNTIL), then r25 evals + E1-E3 diagnostics vs p25o.
cd /home/glenn/projects/neuro
export PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
ctl=models/checkpoints/screen_p25o_7L_chunk512_x_H3add/step_0006104.pt
for r in "$@"; do
  GPU=1 UNTIL="${UNTIL:-2026-10-07 18:55}" bash temp/run_5080_sched.sh $r
  [ -f models/checkpoints/$r/DONE ] || continue
  ck=$(ls models/checkpoints/$r/step_*.pt | tail -1)
  GPU=1 bash temp/run_r25_evals.sh "$r=$ck" 2>&1 | sed "s/^/  r25-evals /" >> logs/gap_queue.log
  for s in xcopy_sweep_eval mem_write_probe multichunk_eval; do
    CUDA_VISIBLE_DEVICES=1 $PY temp/$s.py "$r=$ck" >> logs/e123_diag.log 2>&1
  done
  CUDA_VISIBLE_DEVICES=1 $PY temp/xcopy_sweep_eval.py --contents none "$r=$ck" >> logs/e123_diag.log 2>&1
  echo "[s1 $(date '+%m-%d %H:%M:%S')] [$r] evals + E1-E3 DONE (logs/e123_diag.log)" >> logs/gap_queue.log
done
