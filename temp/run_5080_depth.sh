#!/bin/bash
# run_5080_depth.sh -- 30M depth-vs-width screen on the 5080 (2026-09-25).
# Treatment L4 d832 N1664 x seed {0,1}; control = the already-done screen_bs512_s{0,1}_30m.
# Then recall_eval --replay split --B 32 on all 4 endpoints.
# RESUMABLE like run_5080_width.sh (per-run auto-resume; .done.d markers).
# LAUNCH: Bash run_in_background -> wsl.exe bash -lc "bash temp/run_5080_depth.sh"
# WATCH:  tail -f /home/glenn/projects/neuro/logs/depth_screen.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1          # 5080 ONLY -- never the 5090
export PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/depth_screen.log
DONE=logs/depth_screen.done.d
mkdir -p "$DONE"
exec 9>logs/depth_screen.lock; flock -n 9 || { echo "already running"; exit 1; }

say(){ echo "[depth $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_stage(){ # $1=run_name $2=config
  local tag=$1 cfg=$2 ec
  if [ -f "$DONE/$tag" ]; then say "[$tag] already complete -- skip"; return 0; fi
  say "[$tag] START/RESUME $cfg"
  "$PY" src/train_gpu.py --config "$cfg" --device cuda >> "$LOG" 2>&1
  ec=$?
  if [ "$ec" -eq 0 ]; then touch "$DONE/$tag"; say "[$tag] COMPLETE"; return 0; fi
  say "[$tag] STOPPED exit=$ec"; return "$ec"
}

say "=== depth 30M screen START/RESUME ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"
run_stage screen_depth_L4_s0_30m configs/screen_depth_L4_s0_30m.yaml || exit 1
run_stage screen_depth_L4_s1_30m configs/screen_depth_L4_s1_30m.yaml || exit 1

if [ ! -f "$DONE/D" ]; then
  say "[D] recall_eval --replay split (B=32)"
  "$PY" scripts/recall_eval.py --runs screen_bs512_s0_30m,screen_bs512_s1_30m,screen_depth_L4_s0_30m,screen_depth_L4_s1_30m \
    --Ns 4,8,16,32,64 --B 32 --replay split >> "$LOG" 2>&1 && touch "$DONE/D" || say "[D] recall_eval failed"
fi
say "=== depth 30M screen DONE ==="
