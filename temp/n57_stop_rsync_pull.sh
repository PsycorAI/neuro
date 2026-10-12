#!/bin/bash
# Neuro 57: stop the slow rsync pull of the pinned checkpoint (the Drive pull replaces it), remove its partial file,
# and unpin the checkpoint on the instance. SIGTERM, exact patterns, verify.
cd /home/glenn/projects/neuro || exit 1
pkill -TERM -f '^bash temp/n57_pin.sh pull$'; sleep 1
pkill -TERM -f '^rsync -a --partial --inplace .*n57_pin/step_'; sleep 3
pgrep -af 'n57_pin.sh pull|n57_pin/step_' | grep -v n57_stop | cut -c1-100 || echo "rsync pull stopped"
rm -f models/checkpoints/v4_cloud_pin/dl_step_*.part
timeout 60 bash temp/n57_pin.sh unpin
ls -la models/checkpoints/v4_cloud_pin/ | tail -n 2
