#!/bin/bash
# pull_ckpts.sh <ssh-host> <ssh-port> [runs] -- run at home under setsid (2026-10-04 draft). Every 2 h copies the
# newest checkpoint and the logs from the Vast instance, so losing the host costs at most ~2 h of training.
# 2026-10-11 (Neuro 55): (1) BOTH phases by default (it used to follow v4_cloud only and quit at its DONE, so no
# conversion checkpoint ever came home); (2) keeps only the newest KEEP local copies per run (a 1B checkpoint is
# ~10 GB: 4.5 days of pulls would fill G: and take WSL down with it); (3) quits after the pull that FOLLOWS the last
# run's DONE marker, so the final checkpoint is never missed.
H=${1:?ssh host}; P=${2:?ssh port}; RUNS=${3:-"v4_cloud v4_cloud_conv"}; KEEP=${KEEP:-2}
SSH="ssh -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
L=/home/glenn/projects/neuro/logs/pull_ckpts.log
while :; do
  fin=$($SSH root@$H "test -f /workspace/neuro/models/checkpoints/${RUNS##* }/DONE && echo 1")
  for RUN in $RUNS; do
    DST=/home/glenn/projects/neuro/models/checkpoints/$RUN
    last=$($SSH root@$H "ls /workspace/neuro/models/checkpoints/$RUN/step_*.pt 2>/dev/null | tail -1")
    [ -n "$last" ] || continue
    mkdir -p $DST
    if [ ! -f $DST/$(basename $last) ]; then
      rsync -a --partial -e "$SSH" root@$H:$last $DST/ && \
        echo "[pull $(date '+%m-%d %H:%M')] $RUN $(basename $last)" >> $L
    fi
    ls $DST/step_*.pt 2>/dev/null | head -n -$KEEP | xargs -r rm -f
  done
  rsync -a -e "$SSH" root@$H:/workspace/neuro/logs/ /home/glenn/projects/neuro/logs/cloud/ 2>/dev/null
  [ "$fin" = 1 ] && break
  sleep ${EVERY:-7200}
done
echo "[pull $(date '+%m-%d %H:%M')] ${RUNS##* } DONE, final pull complete" >> $L
