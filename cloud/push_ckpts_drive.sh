#!/bin/bash
# push_ckpts_drive.sh -- 2026-10-11 (Neuro 55). NOT RUN YET (needs rclone + Glenn's one-time authorisation).
# Off-instance copies of the newest FULL checkpoints on Glenn's Google Drive (Glenn 10-11: keep the last 3; his home
# line is 19 Mbps = 70+ min per 10 GB checkpoint, so Drive is the fast, always-on backup and the home PC pulls the
# newest one separately with cloud/pull_ckpts.sh). Runs ON the instance, under setsid. Every EVERY seconds (default
# 2 h) it copies the newest checkpoint of each phase and the logs to <REMOTE>:<DIR>/ with rclone at datacenter speed.
#
# One-time setup, done by Glenn (the script never sees a password; the token stays in rclone's own config file):
#   on the instance:   apt-get install -y rclone && rclone config
#       n) new remote | name: gdrive | storage: drive | client id / secret: empty | scope: drive.file
#       (drive.file = rclone can only see and change files it created itself -- it cannot touch the corpus folder)
#       "Use auto config?" n  -> it prints an "rclone authorize ..." command: run that on the home PC (a browser
#       opens for the Google login), paste the result back on the instance.
#   check:             rclone lsd gdrive:
#   start:             cd /workspace/neuro && HARD_DELETE=1 setsid -f bash cloud/push_ckpts_drive.sh < /dev/null > /dev/null 2>&1
#
# What is kept: the newest KEEP (default 3) checkpoints of the phase that is running, and only the FINAL checkpoint of
# a phase that is DONE. A 1B checkpoint with optimizer state is ~10 GB -> up to ~30 GB during the dense phase and
# ~40 GB during the conversion (1 dense + 3 conversion).
# Older copies: by default rclone moves them to the Drive TRASH, which still counts toward the quota until the trash is
# emptied, so the Drive fills up anyway. HARD_DELETE=1 (Glenn's choice, set on the command line above) deletes the
# superseded copies for good. Either way only files this script uploaded can be affected (drive.file scope).
# WEIGHTS_ONLY=1 uploads the model without the optimizer state (~4.4 GB; a restart from it begins with a fresh
# optimizer, as the conversion phase does anyway).
REMOTE=${REMOTE:-gdrive}; DIR=${DIR:-neuro_v4_ckpts}; EVERY=${EVERY:-7200}; KEEP=${KEEP:-3}; RUNS=${RUNS:-"v4_cloud v4_cloud_conv"}
cd ${ROOT:-/workspace/neuro} || exit 1
PY=${PY:-/workspace/venv/bin/python}; L=logs/push_ckpts_drive.log
TRASH="--drive-use-trash=true"; [ "${HARD_DELETE:-0}" = 1 ] && TRASH="--drive-use-trash=false"
rclone lsd $REMOTE: > /dev/null 2>&1 || { echo "rclone remote '$REMOTE' is not configured -- see the header of this script"; exit 1; }
while :; do
  fin=0; [ -f models/checkpoints/${RUNS##* }/DONE ] && fin=1
  for RUN in $RUNS; do
    ck=models/checkpoints/$RUN
    rm -f $ck/.push_*                                              # left over from an upload that was cut off
    last=$(ls $ck/step_*.pt 2>/dev/null | tail -1); [ -n "$last" ] || continue
    name=$(basename $last)
    if ! rclone lsf $REMOTE:$DIR/$RUN/ 2>/dev/null | grep -qx "$name"; then        # not on Drive yet
      # A hard link under another name: the trainer's keep_last rotation cannot delete the file in mid-upload, and the
      # trainer's own resume / cleanup globs (step_*.pt, step_*.pt.tmp) do not match it.
      if [ "${WEIGHTS_ONLY:-0}" = 1 ]; then
        $PY -c "import torch,sys; s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); torch.save({k:v for k,v in s.items() if k not in ('opt_states','opt')}, sys.argv[2])" $last $ck/.push_$name || continue
      else
        ln $last $ck/.push_$name || continue
      fi
      if rclone copyto $ck/.push_$name $REMOTE:$DIR/$RUN/$name --drive-chunk-size 256M; then
        echo "[push $(date '+%m-%d %H:%M')] $RUN $name ($(du -h $ck/.push_$name | cut -f1))" >> $L
      else
        echo "[push $(date '+%m-%d %H:%M')] $RUN $name FAILED" >> $L
      fi
      rm -f $ck/.push_$name
    fi
    keep=$KEEP; [ -f $ck/DONE ] && keep=1                          # a finished phase keeps only its final checkpoint
    rclone lsf $REMOTE:$DIR/$RUN/ 2>/dev/null | grep -E '^step_[0-9]+\.pt$' | sort | head -n -$keep | while read -r f; do
      rclone deletefile $TRASH $REMOTE:$DIR/$RUN/$f && echo "[push $(date '+%m-%d %H:%M')] removed $RUN $f ($TRASH)" >> $L
    done
  done
  rclone copy logs $REMOTE:$DIR/logs --include "*.log" --max-size 50M > /dev/null 2>&1
  [ $fin = 1 ] && break
  sleep $EVERY
done
echo "[push $(date '+%m-%d %H:%M')] ${RUNS##* } DONE, final push complete" >> $L
