#!/bin/bash
# n57_drive_pull.sh <run> <step_file> -- 2026-10-12 (Neuro 57). Pull one checkpoint from Glenn's Drive
# (gdrive:neuro_v4_ckpts/<run>/<file>, uploaded by cloud/push_ckpts_drive.sh) to models/checkpoints/<run>_pin/.
# Size-checked; one DONE / ALERT. Log: logs/n57_drive_pull.log
# Launch: setsid -f bash temp/n57_drive_pull.sh v4_cloud step_0046500.pt < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
R=${1:?run}; F=${2:?file}; D=models/checkpoints/${R}_pin; mkdir -p $D
timeout 21600 rclone copyto gdrive:neuro_v4_ckpts/$R/$F $D/$F --drive-chunk-size 64M --retries 5 --stats 60s --stats-one-line -v > logs/n57_drive_pull.log 2>&1
want=$(rclone size --json gdrive:neuro_v4_ckpts/$R/$F 2>/dev/null | sed -n 's/.*"bytes":\([0-9]*\).*/\1/p'); have=$(stat -c %s $D/$F 2>/dev/null)
if [ -n "$want" ] && [ "$want" = "$have" ]; then bash temp/alert.sh DONE "v4 checkpoint $F is home from Drive: $D/ ($have bytes)"
else bash temp/alert.sh ALERT "Drive pull of $F: have ${have:-0} of ${want:-?} bytes (logs/n57_drive_pull.log)"; fi
