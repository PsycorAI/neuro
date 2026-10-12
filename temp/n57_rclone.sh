#!/bin/bash
# n57_rclone.sh setup|pinup|start|status -- 2026-10-12 (Neuro 57; Glenn's explicit OK to copy his rclone config to the
# instance). setup = install rclone on the instance, copy the config (mode 600), test.  pinup = upload the pinned
# checkpoint (/workspace/n57_pin) to Drive now.  start = start cloud/push_ckpts_drive.sh.  status = logs + Drive listing.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
CONF=$(rclone config file | tail -n 1)
case "$1" in
  setup)  $SSH root@$H 'command -v rclone > /dev/null || (curl -s https://rclone.org/install.sh | bash > /tmp/rclone_install.log 2>&1); rclone version | head -n 1; mkdir -p /root/.config/rclone' 2>/dev/null
          scp -q -P $P -o BatchMode=yes "$CONF" root@$H:/root/.config/rclone/rclone.conf && $SSH root@$H 'chmod 600 /root/.config/rclone/rclone.conf; rclone lsd gdrive: 2>&1 | tail -n 3; rclone about gdrive: 2>&1 | grep -E "Free|Used"' 2>/dev/null ;;
  pinup)  $SSH root@$H 'cd /workspace/neuro; f=$(ls /workspace/n57_pin/step_*.pt); setsid -f bash -c "t0=\$(date +%s); rclone --drive-chunk-size 256M copyto $f gdrive:neuro_v4_ckpts/v4_cloud/$(basename $f) > logs/n57_pinup.log 2>&1; echo \"[pinup] exit=\$? \$(( \$(date +%s) - t0 )) s\" >> logs/n57_pinup.log" < /dev/null > /dev/null 2>&1; echo "upload of $(basename $f) started"' 2>/dev/null ;;
  start)  scp -q -P $P -o BatchMode=yes cloud/push_ckpts_drive.sh root@$H:/workspace/neuro/cloud/ && $SSH root@$H 'cd /workspace/neuro && bash -n cloud/push_ckpts_drive.sh && HARD_DELETE=1 setsid -f bash cloud/push_ckpts_drive.sh < /dev/null > /dev/null 2>&1; sleep 5; pgrep -af "^bash cloud/push_ckpts_drive.sh" | cut -c1-60' 2>/dev/null ;;
  status) $SSH root@$H 'cd /workspace/neuro; cat logs/n57_pinup.log 2>/dev/null | tail -n 2; tail -n 5 logs/push_ckpts_drive.log 2>/dev/null; rclone lsl gdrive:neuro_v4_ckpts 2>/dev/null; pgrep -af "^bash cloud/push_ckpts_drive.sh|rclone " | cut -c1-70; grep -E "^step" logs/v4_cloud.log | tail -n 1 | cut -c1-80' 2>/dev/null ;;
esac
