#!/bin/bash
# run_e123.sh -- 2026-10-06 (Neuro 52). Diagnostics E1-E3 (docs/next_experiments_2026_10_06.md) on the 5080.
# Usage: setsid -f bash temp/run_e123.sh [SMOKE]   -> logs/e123_diag.log (+ summary lines in logs/gap_queue.log)
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/e123_diag.log
CK=models/checkpoints
RUNS="r25o=$CK/r25o_13L_H6only_150m/step_0012208.pt r25v=$CK/r25v_13L_H6_xcopy_150m/step_0012208.pt r25w=$CK/r25w_13L_spk_kd_sc15_150m/step_0017090.pt"
if [ "$1" = "SMOKE" ]; then
  R="r25o=$CK/r25o_13L_H6only_150m/step_0012208.pt"
  $PY temp/xcopy_sweep_eval.py --reps 1 $R 2>&1 | tail -n 6
  $PY temp/mem_write_probe.py --reps 1 $R 2>&1 | tail -n 14
  $PY temp/multichunk_eval.py --reps 1 $R 2>&1 | tail -n 6
  exit 0
fi
echo "[e123 $(date '+%m-%d %H:%M:%S')] START" >> $LOG
$PY temp/xcopy_sweep_eval.py $RUNS >> $LOG 2>&1
echo "[e123 $(date '+%m-%d %H:%M:%S')] E1 done" >> $LOG
$PY temp/mem_write_probe.py $RUNS >> $LOG 2>&1
echo "[e123 $(date '+%m-%d %H:%M:%S')] E2 done" >> $LOG
$PY temp/multichunk_eval.py $RUNS >> $LOG 2>&1
echo "[e123 $(date '+%m-%d %H:%M:%S')] E3 DONE" >> $LOG
echo "[g1 $(date '+%m-%d %H:%M:%S')] E1-E3 diagnostics (xcopy sweep, memory write probe, multi-chunk) DONE -> logs/e123_diag.log" >> logs/gap_queue.log
