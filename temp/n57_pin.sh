#!/bin/bash
# n57_pin.sh pin|itest|pull|status|unpin -- 2026-10-12 (Neuro 57, Glenn OK). One v4 checkpoint PINNED on the instance
# (hard link in /workspace/n57_pin/, 10 GB once the trainer prunes its own copy) so that (a) the instance tests and
# (b) the ~4 h download to this PC can finish.  pull = rsync to models/checkpoints/v4_cloud_pin/ (resumable; run it
# under setsid; log logs/n57_pull.log).  unpin = remove the link when the download is verified.
cd /home/glenn/projects/neuro || exit 1
. temp/n56_host.env
case "$1" in
  pin)    $SSH root@$H 'cd /workspace/neuro && mkdir -p /workspace/n57_pin && rm -f /workspace/n57_pin/*.pt && ck=$(ls models/checkpoints/v4_cloud/step_*.pt | tail -1) && ln $ck /workspace/n57_pin/ && ls -la /workspace/n57_pin | tail -n 1; df -h /workspace | tail -n 1' 2>/dev/null ;;
  itest)  scp -q -P $P -o BatchMode=yes temp/n57_sample.py temp/n57_itest_remote.sh root@$H:/workspace/neuro/temp/ || exit 1
          $SSH root@$H 'cd /workspace/neuro && setsid -f bash temp/n57_itest_remote.sh $(ls /workspace/n57_pin/step_*.pt) < /dev/null > /dev/null 2>&1; echo itest started' 2>/dev/null ;;
  pull)   mkdir -p models/checkpoints/v4_cloud_pin
          f=$($SSH root@$H 'ls /workspace/n57_pin/step_*.pt' 2>/dev/null)
          echo "[pull $(date '+%m-%d %H:%M')] start $f" >> logs/n57_pull.log
          until rsync -a --partial --inplace -e "$SSH" root@$H:$f models/checkpoints/v4_cloud_pin/dl_$(basename $f).part; do sleep 60; done
          want=$($SSH root@$H "stat -c %s $f" 2>/dev/null); have=$(stat -c %s models/checkpoints/v4_cloud_pin/dl_$(basename $f).part)
          if [ "$want" = "$have" ]; then mv models/checkpoints/v4_cloud_pin/dl_$(basename $f).part models/checkpoints/v4_cloud_pin/$(basename $f)
            echo "[pull $(date '+%m-%d %H:%M')] complete $(basename $f) $have bytes" >> logs/n57_pull.log; bash temp/alert.sh DONE "v4 checkpoint $(basename $f) is home: models/checkpoints/v4_cloud_pin/ ($have bytes)"
          else bash temp/alert.sh ALERT "v4 checkpoint pull: size mismatch $have vs $want"; fi ;;
  status) $SSH root@$H 'cat /workspace/neuro/logs/n57_itest.log | cut -c1-700; ls -la /workspace/n57_pin | tail -n 1; grep -E "^step" /workspace/neuro/logs/v4_cloud.log | tail -n 1 | cut -c1-80' 2>/dev/null
          ls -la --time-style=+%H:%M models/checkpoints/v4_cloud_pin/ 2>/dev/null | tail -n 2; tail -n 2 logs/n57_pull.log 2>/dev/null ;;
  unpin)  $SSH root@$H 'rm -f /workspace/n57_pin/*.pt; ls -la /workspace/n57_pin' 2>/dev/null ;;
esac
