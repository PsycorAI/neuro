#!/bin/bash
# n58_pull_logs.sh -- 2026-10-12 (Neuro 58). Logs-only replacement of cloud/pull_ckpts.sh (its checkpoint rsync could
# never finish; checkpoints now come from Drive, temp/n57_drive_pull.sh). Every 30 min copies the instance's logs to
# logs/cloud/; stops when the conversion run's DONE marker exists.
# Launch: setsid -f bash temp/n58_pull_logs.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
while :; do
  fin=$($SSH root@$H "test -f /workspace/neuro/models/checkpoints/v4_cloud_conv/DONE && echo 1")
  timeout 900 rsync -a -e "$SSH" root@$H:/workspace/neuro/logs/ logs/cloud/ 2>/dev/null
  [ "$fin" = 1 ] && break
  sleep 1800
done
