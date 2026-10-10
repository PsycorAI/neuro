#!/bin/bash
# run_v4.sh -- Vast on-start command + manual launcher (2026-10-04 draft). train_gpu auto-resumes from the latest
# step_*.pt in models/checkpoints/<run_name>, so an interrupted instance continues where it stopped. Retries on a crash
# (max 5 in a row), stops when the run is DONE.
# 2026-10-09: two phases (RC3). Dense -> temp/mk_spike_seed.py seeds the conversion run from the last dense ckpt (own
# run dir: the dense ckpt stays as the fallback demo model) -> conversion.
# Usage: bash cloud/run_v4.sh [dense_run=v4_cloud] [conv_run=v4_cloud_conv]
# Home rehearsal (temp/queue_5080_reh.sh) overrides ROOT / PY / HF_HOME / PYTORCH_CUDA_ALLOC_CONF and the run names.
cd ${ROOT:=/workspace/neuro}
PY=${PY:-/workspace/venv/bin/python}
DENSE=${1:-v4_cloud}; CONV=${2:-v4_cloud_conv}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1   # Llama-3.2-1B teacher: downloaded once by Glenn (gated), no token here
say(){ echo "[run_v4 $(date '+%m-%d %H:%M:%S')] $*" >> logs/run_v4.log; }
phase(){ CFG=configs/$1.yaml; RUN=$1; fails=0
  while [ ! -f models/checkpoints/$RUN/DONE ]; do
    say "start/resume $RUN"
    $PY src/train_gpu.py --config $CFG --device cuda >> logs/$RUN.log 2>&1
    rc=$?
    say "$RUN exit=$rc"
    if [ $rc = 0 ]; then touch models/checkpoints/$RUN/DONE; break; fi
    # exit 2 = val_ppl tripwire, 3 = nan guard: a ckpt was saved; retrying would only diverge again -> stop for a human
    if [ $rc = 2 ] || [ $rc = 3 ]; then say "$RUN stopped by a guard (exit $rc), NOT retrying"; exit 1; fi
    fails=$((fails + 1)); [ $fails -ge 5 ] && { say "5 failures in a row, giving up"; exit 1; }
    sleep 60
  done; }
phase $DENSE
if ! ls models/checkpoints/$CONV/step_*.pt >/dev/null 2>&1; then
  $PY temp/mk_spike_seed.py "$(ls models/checkpoints/$DENSE/step_*.pt | tail -1)" models/checkpoints/$CONV \
      configs/$CONV.yaml 1.0 >> logs/run_v4.log 2>&1 || { say "seed FAILED"; exit 1; }
fi
phase $CONV
say "ALL DONE"
