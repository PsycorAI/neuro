#!/bin/bash
# n58_drive_next.sh [after=step_0055500.pt] -- 2026-10-12 (Neuro 58, Glenn: "the most recent instance checkpoint").
# Polls Drive every 2 min (max 3 h) for a v4_cloud checkpoint newer than <after> whose upload is complete (full size),
# then pulls it with temp/n57_drive_pull.sh (one DONE / ALERT from there).
# Launch: setsid -f bash temp/n58_drive_next.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
AFTER=${1:-step_0055500.pt}; end=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$end" ]; do
  new=$(rclone lsl gdrive:neuro_v4_ckpts/v4_cloud/ 2>/dev/null | awk '$1 == 9749773355 {print $4}' | grep -E '^step_[0-9]+\.pt$' | sort | tail -n 1)
  if [ -n "$new" ] && [ "$new" \> "$AFTER" ]; then exec bash temp/n57_drive_pull.sh v4_cloud $new; fi
  sleep 120
done
bash temp/alert.sh ALERT "n58_drive_next: no checkpoint newer than $AFTER on Drive after 3 h"
