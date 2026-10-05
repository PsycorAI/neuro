#!/bin/bash
# pull_ckpts.sh <ssh-host> <ssh-port> [run_name] -- run at home under setsid (2026-10-04 draft). Every 2 h copies the
# newest checkpoint and the logs from the Vast instance, so losing the host costs at most ~2 h of training.
H=${1:?ssh host}; P=${2:?ssh port}; RUN=${3:-v4_cloud}
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
DST=/home/glenn/projects/neuro/models/checkpoints/$RUN
mkdir -p $DST
while :; do
  last=$($SSH root@$H "ls /workspace/neuro/models/checkpoints/$RUN/step_*.pt 2>/dev/null | tail -1")
  if [ -n "$last" ]; then
    rsync -a --partial -e "$SSH" root@$H:$last $DST/ && \
      echo "[pull $(date '+%m-%d %H:%M')] $(basename $last)" >> /home/glenn/projects/neuro/logs/pull_ckpts.log
  fi
  rsync -a -e "$SSH" root@$H:/workspace/neuro/logs/ /home/glenn/projects/neuro/logs/cloud/ 2>/dev/null
  [ $($SSH root@$H "test -f /workspace/neuro/models/checkpoints/$RUN/DONE && echo 1") = 1 ] 2>/dev/null && break
  sleep 7200
done
echo "[pull $(date '+%m-%d %H:%M')] run DONE, final pull complete" >> /home/glenn/projects/neuro/logs/pull_ckpts.log
