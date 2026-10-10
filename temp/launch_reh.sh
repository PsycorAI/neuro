#!/bin/bash
# launch_reh.sh -- 2026-10-09 (Neuro 54). One-shot: syntax-check the cloud scripts, launch the 5080 dress rehearsal.
cd /home/glenn/projects/neuro
for f in cloud/run_v4.sh cloud/pull_corpus.sh cloud/push_from_home.sh temp/queue_5080_reh.sh; do bash -n $f < /dev/null || { echo "BAD $f"; exit 1; }; done
pgrep -f "^bash temp/queue_5080_reh.sh" > /dev/null && { echo "rehearsal already running"; exit 0; }
ls models/checkpoints | grep -q '^v4_reh' && { echo "v4_reh checkpoints exist: not launching"; exit 1; }
setsid -f bash temp/queue_5080_reh.sh < /dev/null > /dev/null 2>&1
sleep 5
pgrep -af "^bash temp/queue_5080_reh" | cut -c1-60
tail -n 2 logs/alerts.log
