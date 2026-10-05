#!/bin/bash
# run_v4.sh -- Vast on-start command + manual launcher (2026-10-04 draft). train_gpu auto-resumes from the latest
# step_*.pt in models/checkpoints/<run_name>, so an interrupted instance continues where it stopped. Retries on a crash
# (max 5 in a row), stops when the run is DONE.
cd /workspace/neuro
CFG=configs/v4_cloud.yaml
RUN=$(grep '^run_name:' $CFG | awk '{print $2}')
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fails=0
while [ ! -f models/checkpoints/$RUN/DONE ]; do
  echo "[run_v4 $(date '+%m-%d %H:%M:%S')] start/resume $RUN" >> logs/run_v4.log
  /workspace/venv/bin/python src/train_gpu.py --config $CFG --device cuda >> logs/$RUN.log 2>&1
  rc=$?
  echo "[run_v4 $(date '+%m-%d %H:%M:%S')] exit=$rc" >> logs/run_v4.log
  if [ $rc = 0 ]; then touch models/checkpoints/$RUN/DONE; break; fi
  fails=$((fails + 1)); [ $fails -ge 5 ] && { echo "[run_v4] 5 failures in a row, giving up" >> logs/run_v4.log; exit 1; }
  sleep 60
done
