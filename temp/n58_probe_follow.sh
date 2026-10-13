#!/bin/bash
# n58_probe_follow.sh [run] [gpu] -- 2026-10-12 (Neuro 58). Evaluates each new checkpoint of the home probe on the
# 5080 (small batch): cross-chunk haystack + spike diagnostics. The checkpoint is hard-linked first (the trainer keeps
# only 2). One DONE alert per evaluated checkpoint; ends when the probe driver is gone and nothing new is left.
# 40 min limit per eval. Log: logs/n58_probe_evals.log
# Launch: setsid -f bash temp/n58_probe_follow.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
RUN=${1:-probe_v4_xchunk}; D=models/checkpoints/$RUN; E=models/checkpoints/${RUN}_eval; LOG=logs/n58_probe_evals.log
PY=/home/glenn/projects/bdh/venv/bin/python
export CUDA_VISIBLE_DEVICES=${2:-1} PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HAYSTACK_B=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $E; seen=" step_0046500.pt "
while :; do
  new=$(ls $D/step_*.pt 2>/dev/null | tail -n 1); f=$(basename "$new" 2>/dev/null)
  if [ -n "$new" ] && [ "${seen#* $f }" = "$seen" ]; then
    seen="$seen$f "; ln -f $new $E/$f 2>/dev/null || { sleep 20; continue; }
    echo "== $f ($(date '+%m-%d %H:%M'))" >> $LOG
    x=$(timeout 2400 $PY temp/xchunk_haystack_eval.py p=$E/$f 2>&1 | grep -E 'C=|rror|Killed' | cut -c1-500); echo "$x" >> $LOG
    timeout 2400 $PY temp/n58_spike_diag.py --B 1 p=$E/$f 2>&1 | grep -E 'rate|per |code|by chunk|rror|Killed' | cut -c1-300 >> $LOG
    v=$(grep -E 'eval\]' logs/$RUN.log | tail -n 1 | cut -c1-110)
    bash temp/alert.sh DONE "n58 probe $f: $(echo $x | cut -c1-260) |$v"
    rm -f $E/$f
  elif ! pgrep -f "^bash temp/n58_probe.sh" > /dev/null; then break
  else sleep 60; fi
done
