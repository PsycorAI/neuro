#!/bin/bash
# push_ckpts_drive.sh -- 2026-10-11 (Neuro 55). NOT RUN YET (needs rclone + Glenn's one-time authorisation).
# Glenn: his download line is 19 Mbps = 70+ min per 10 GB checkpoint, so the off-instance copies go to his Google
# Drive instead of being pulled home (cloud/pull_ckpts.sh). Runs ON the instance, under setsid. Every EVERY seconds
# (default 6 h) it copies the newest checkpoint of each phase and the logs to <REMOTE>:<DIR>/ with rclone, at
# datacenter speed. Purpose: a restart point if the host is lost for good -- the instance's own disk already holds a
# checkpoint every ~25 min and survives an interruption.
#
# One-time setup, done by Glenn (the script never sees a password; the token stays in rclone's own config file):
#   on the instance:   apt-get install -y rclone && rclone config
#       n) new remote | name: gdrive | storage: drive | client id / secret: empty | scope: drive.file
#       (drive.file = rclone can only see and change files it created itself, nothing else in the Drive)
#       "Use auto config?" n  -> it prints an "rclone authorize ..." command: run that on the home PC (a browser
#       opens for the Google login), paste the result back on the instance.
#   check:             rclone lsd gdrive:
#   start:             cd /workspace/neuro && setsid -f bash cloud/push_ckpts_drive.sh < /dev/null > /dev/null 2>&1
#
# Drive space: a dense 1B checkpoint is ~10 GB (weights + optimizer state). Only the newest KEEP (default 2) per phase
# are kept; rclone moves the older ones to the Drive TRASH, which still counts toward the quota until the trash is
# emptied by hand. WEIGHTS_ONLY=1 uploads the model without the optimizer state (~4.4 GB; a restart from it begins
# with a fresh optimizer, as the conversion phase does anyway).
REMOTE=${REMOTE:-gdrive}; DIR=${DIR:-neuro_v4_ckpts}; EVERY=${EVERY:-21600}; KEEP=${KEEP:-2}; RUNS=${RUNS:-"v4_cloud v4_cloud_conv"}
cd ${ROOT:-/workspace/neuro} || exit 1
PY=${PY:-/workspace/venv/bin/python}; L=logs/push_ckpts_drive.log
rclone lsd $REMOTE: > /dev/null 2>&1 || { echo "rclone remote '$REMOTE' is not configured -- see the header of this script"; exit 1; }
while :; do
  fin=0; [ -f models/checkpoints/${RUNS##* }/DONE ] && fin=1
  for RUN in $RUNS; do
    ck=models/checkpoints/$RUN
    rm -f $ck/.push_*                                              # left over from an upload that was cut off
    last=$(ls $ck/step_*.pt 2>/dev/null | tail -1); [ -n "$last" ] || continue
    name=$(basename $last)
    rclone lsf $REMOTE:$DIR/$RUN/ 2>/dev/null | grep -qx "$name" && continue          # already on Drive
    # A hard link under another name: the trainer's keep_last rotation cannot delete the file in mid-upload, and the
    # trainer's own resume / cleanup globs (step_*.pt, step_*.pt.tmp) do not match it.
    if [ "${WEIGHTS_ONLY:-0}" = 1 ]; then
      $PY -c "import torch,sys; s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); torch.save({k:v for k,v in s.items() if k not in ('opt_states','opt')}, sys.argv[2])" $last $ck/.push_$name || continue
    else
      ln $last $ck/.push_$name || continue
    fi
    if rclone copyto $ck/.push_$name $REMOTE:$DIR/$RUN/$name --drive-chunk-size 256M; then
      echo "[push $(date '+%m-%d %H:%M')] $RUN $name ($(du -h $ck/.push_$name | cut -f1))" >> $L
      rclone lsf $REMOTE:$DIR/$RUN/ 2>/dev/null | grep -E '^step_[0-9]+\.pt$' | sort | head -n -$KEEP | while read -r f; do
        rclone deletefile $REMOTE:$DIR/$RUN/$f                     # to the Drive trash (rclone's default), not a hard delete
      done
    else
      echo "[push $(date '+%m-%d %H:%M')] $RUN $name FAILED" >> $L
    fi
    rm -f $ck/.push_$name
  done
  rclone copy logs $REMOTE:$DIR/logs --include "*.log" --max-size 50M > /dev/null 2>&1
  [ $fin = 1 ] && break
  sleep $EVERY
done
echo "[push $(date '+%m-%d %H:%M')] ${RUNS##* } DONE, final push complete" >> $L
