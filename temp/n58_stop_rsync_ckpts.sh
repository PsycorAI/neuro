#!/bin/bash
# Neuro 58 (Glenn OK 10-12): stop the checkpoint rsync pull for good, delete its truncated files, start the logs-only pull.
cd /home/glenn/projects/neuro || exit 1
pkill -TERM -f '^bash cloud/pull_ckpts.sh'; sleep 1
pkill -TERM -f '^rsync -a --partial -e ssh .*checkpoints/v4_cloud/step_'; sleep 3
pgrep -af '^bash cloud/pull_ckpts.sh|^rsync -a --partial' | cut -c1-100 || echo "pull_ckpts + rsync stopped"
D=models/checkpoints/v4_cloud
rm -f $D/step_0018000.pt $D/TRUNCATED_step_0007500.partial $D/.step_0040500.pt.*
ls -la $D
setsid -f bash temp/n58_pull_logs.sh < /dev/null > /dev/null 2>&1
sleep 2; pgrep -af '^bash temp/n58_pull_logs.sh'
df -h /home | tail -n 1
