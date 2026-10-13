#!/bin/bash
# n58_probe.sh [run] -- 2026-10-12 (Neuro 58, Glenn OK 10-12 "use the 5090 for the probe"). Home probe on the 5090:
# resume the v4 cloud checkpoint step 46,500 with cross-chunk recall training switched on now (configs/<run>.yaml from
# temp/n58_mk_probe.py). Starts after the battery (logs/n58_battery.log "[battery] done"). 14 h limit. One DONE / ALERT.
# Evals of its checkpoints: temp/n58_probe_follow.sh (5080). Log: logs/<run>.log
# Launch: setsid -f bash temp/n58_probe.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
RUN=${1:-probe_v4_xchunk}; SRC=models/checkpoints/v4_cloud_pin/step_0046500.pt
PY=/home/glenn/projects/bdh/venv/bin/python
end=$(( $(date +%s) + 21600 ))
until grep -q '^\[battery\] done' logs/n58_battery.log 2>/dev/null && ! pgrep -f '^bash temp/n58_battery' > /dev/null; do
  [ "$(date +%s)" -ge "$end" ] && { bash temp/alert.sh ALERT "n58 probe: the battery did not finish in 6 h"; exit 1; }
  sleep 30
done
mkdir -p models/checkpoints/$RUN
ls models/checkpoints/$RUN/step_*.pt > /dev/null 2>&1 || ln $SRC models/checkpoints/$RUN/ || exit 1
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
bash temp/alert.sh INFO "n58 probe $RUN started on the 5090 (logs/$RUN.log)"
timeout -s TERM 50400 $PY src/train_gpu.py --config configs/$RUN.yaml --device cuda >> logs/$RUN.log 2>&1
rc=$?
last=$(grep -E '^step' logs/$RUN.log | tail -n 1 | cut -c1-90)
if [ $rc = 124 ] || [ $rc = 0 ]; then bash temp/alert.sh DONE "n58 probe $RUN ended (rc $rc): $last | logs/n58_probe_evals.log"
else bash temp/alert.sh ALERT "n58 probe $RUN died (rc $rc): $last | $(grep -E 'Error|Traceback' logs/$RUN.log | tail -n 1 | cut -c1-140)"; fi
