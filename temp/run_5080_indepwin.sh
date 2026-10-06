#!/bin/bash
# run_5080_indepwin.sh -- 30M independent-windows screen on the 5080 (2026-09-25).
# Waits for the depth screen to finish, then: treatment x seed {0,1}; control =
# the already-done screen_bs512_s{0,1}_30m. Then recall_eval --replay split on all 4.
# WATCH: tail -f /home/glenn/projects/neuro/logs/indepwin_screen.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1          # 5080 ONLY
export PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/indepwin_screen.log
DONE=logs/indepwin_screen.done.d
mkdir -p "$DONE"
exec 9>logs/indepwin_screen.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[indepwin $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for depth screen to finish"
while pgrep -f '[r]un_5080_depth.sh' >/dev/null; do sleep 60; done
say "=== indepwin 30M screen START/RESUME ==="

run_stage(){ # $1=run_name $2=config
  local tag=$1 cfg=$2 ec
  if [ -f "$DONE/$tag" ]; then say "[$tag] already complete -- skip"; return 0; fi
  say "[$tag] START/RESUME $cfg"
  "$PY" src/train_gpu.py --config "$cfg" --device cuda >> "$LOG" 2>&1
  ec=$?
  if [ "$ec" -eq 0 ]; then touch "$DONE/$tag"; say "[$tag] COMPLETE"; return 0; fi
  say "[$tag] STOPPED exit=$ec"; return "$ec"
}
run_stage screen_indepwin_s0_30m configs/screen_indepwin_s0_30m.yaml || exit 1
run_stage screen_indepwin_s1_30m configs/screen_indepwin_s1_30m.yaml || exit 1
if [ ! -f "$DONE/D" ]; then
  say "[D] recall_eval --replay split (B=32)"
  "$PY" scripts/recall_eval.py --runs screen_bs512_s0_30m,screen_bs512_s1_30m,screen_indepwin_s0_30m,screen_indepwin_s1_30m \
    --Ns 4,8,16,32,64 --B 32 --replay split >> "$LOG" 2>&1 && touch "$DONE/D" || say "[D] recall_eval failed"
fi
say "=== indepwin 30M screen DONE ==="
