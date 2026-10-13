#!/bin/bash
# n58_drive_loop.sh [after=step_0055500.pt] -- 2026-10-12 (Neuro 58; Glenn: "keep downloading v4 checkpoints to my
# local PC and test them"). Replaces n58_drive_next.sh (it matched one exact byte size; checkpoints saved after the
# 14:20 fix are 64 bytes larger, so it never fired).
# Forever (until the conversion run's final checkpoint is home): poll Drive every 5 min for a checkpoint newer than the
# last one pulled, in v4_cloud then v4_cloud_conv; a file counts when its size is the same at two polls (upload done).
# Pull it with temp/n57_drive_pull.sh (-> models/checkpoints/<run>_pin/, one DONE / ALERT), keep the newest KEEP (2)
# local copies per run plus step_0046500, then run temp/n58_ckpt_health.sh on it (waits for free RAM by itself).
# Launch: setsid -f bash temp/n58_drive_loop.sh < /dev/null > /dev/null 2>&1
cd /home/glenn/projects/neuro || exit 1
declare -A LAST SIZE; LAST[v4_cloud]=${1:-step_0055500.pt}; LAST[v4_cloud_conv]=step_0000000.pt
while :; do
  for RUN in v4_cloud v4_cloud_conv; do
    line=$(rclone lsl gdrive:neuro_v4_ckpts/$RUN/ 2>/dev/null | awk '$4 ~ /^step_[0-9]+\.pt$/ {print $4, $1}' | sort | tail -n 1)
    set -- $line; f=$1; sz=$2
    [ -n "$f" ] && [ "$f" \> "${LAST[$RUN]}" ] && [ "${sz:-0}" -gt 9000000000 ] || continue
    if [ "${SIZE[$RUN.$f]}" != "$sz" ]; then SIZE[$RUN.$f]=$sz; continue; fi          # size must hold for one more poll
    bash temp/n57_drive_pull.sh $RUN $f
    D=models/checkpoints/${RUN}_pin
    if [ -f $D/$f ]; then
      LAST[$RUN]=$f
      ls $D/step_*.pt | grep -v step_0046500 | head -n -${KEEP:-2} | xargs -r rm -f
      setsid -f bash temp/n58_ckpt_health.sh $D/$f < /dev/null > /dev/null 2>&1
    fi
  done
  [ -f models/checkpoints/v4_cloud_conv_pin/DONE_PULLED ] && break
  sleep 300
done
