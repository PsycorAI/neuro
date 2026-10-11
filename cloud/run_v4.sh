#!/bin/bash
# run_v4.sh -- Vast on-start command + manual launcher (2026-10-04 draft). train_gpu auto-resumes from the latest
# step_*.pt in models/checkpoints/<run_name>, so an interrupted instance continues where it stopped. Retries on a crash
# (max 5 in a row), stops when the run is DONE.
# 2026-10-09: two phases (RC3). Dense -> temp/mk_spike_seed.py seeds the conversion run from the last dense ckpt (own
# run dir: the dense ckpt stays as the fallback demo model) -> conversion.
# Usage: bash cloud/run_v4.sh [dense_run=v4_cloud] [conv_run=v4_cloud_conv]
# Home rehearsal (temp/queue_5080_reh.sh) overrides ROOT / PY / HF_HOME / PYTORCH_CUDA_ALLOC_CONF and the run names.
# 2026-10-11 (Neuro 55):
#  * GO file: on the instance nothing starts until /workspace/neuro/GO exists (touch it at launch). As the on-start
#    command this script also fires after any reboot during setup / bench, and would then train on the placeholder budget.
#  * "5 in a row" was 5 per phase: the counter now resets after a failure-free 10 minutes.
#  * the seed step is redone if it was cut off (SEEDED marker), instead of resuming from a half-written seed checkpoint.
#  * a warning at start if the conversion inputs (KD slice, teacher) are not on the instance yet.
cd ${ROOT:=/workspace/neuro} || exit 1
PY=${PY:-/workspace/venv/bin/python}
DENSE=${1:-v4_cloud}; CONV=${2:-v4_cloud_conv}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1   # Llama-3.2-1B teacher: cloud/get_teacher.sh (gated), no token here
say(){ echo "[run_v4 $(date '+%m-%d %H:%M:%S')] $*" >> logs/run_v4.log; }
if [ "$ROOT" = /workspace/neuro ] && [ ! -f GO ]; then
  echo "[run_v4] no GO file -> not starting. Launch with: touch /workspace/neuro/GO && setsid -f bash cloud/run_v4.sh"; exit 0
fi
kd=$(grep '^data_bin:' configs/$CONV.yaml | awk '{print $2}')
[ -f "$kd" ] || say "WARNING: $kd is not here yet -- the conversion phase needs it (cloud/pull_corpus.sh)"
ls $HF_HOME/hub/models--meta-llama--Llama-3.2-1B/snapshots/*/config.json > /dev/null 2>&1 || \
  say "WARNING: the Llama-3.2-1B teacher is not in $HF_HOME yet -- the conversion phase needs it (cloud/get_teacher.sh)"
phase(){ CFG=configs/$1.yaml; RUN=$1; fails=0
  while [ ! -f models/checkpoints/$RUN/DONE ]; do
    say "start/resume $RUN"; t0=$(date +%s)
    $PY src/train_gpu.py --config $CFG --device cuda >> logs/$RUN.log 2>&1
    rc=$?
    say "$RUN exit=$rc"
    if [ $rc = 0 ]; then touch models/checkpoints/$RUN/DONE; break; fi
    # exit 2 = val_ppl tripwire, 3 = nan guard: a ckpt was saved; retrying would only diverge again -> stop for a human
    if [ $rc = 2 ] || [ $rc = 3 ]; then say "$RUN stopped by a guard (exit $rc), NOT retrying"; exit 1; fi
    [ $(( $(date +%s) - t0 )) -ge 600 ] && fails=0
    fails=$((fails + 1)); [ $fails -ge 5 ] && { say "5 failures in a row, giving up"; exit 1; }
    sleep 60
  done; }
phase $DENSE
if [ ! -f models/checkpoints/$CONV/SEEDED ] && [ "$(ls models/checkpoints/$CONV/step_*.pt 2>/dev/null | wc -l)" -le 1 ]; then
  rm -f models/checkpoints/$CONV/step_*.pt                   # a seed write that was cut off
  $PY temp/mk_spike_seed.py "$(ls models/checkpoints/$DENSE/step_*.pt | tail -1)" models/checkpoints/$CONV \
      configs/$CONV.yaml 1.0 >> logs/run_v4.log 2>&1 || { say "seed FAILED"; exit 1; }
  touch models/checkpoints/$CONV/SEEDED
fi
phase $CONV
say "ALL DONE"
