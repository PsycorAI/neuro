#!/bin/bash
# run_5080_lifexact.sh -- 2026-09-27 30M screen: exact sequential LIF kernel (lif_exact:true)
# vs the K=10 FPT lif_parallel. Control = the already-done screen_indepwin_s{0,1}_30m
# (identical config except lif_exact). Waits for the profiler timing run, then trains
# treatment x seed {0,1}, then recall (split-replay = valid, no attention at 30M; + single-pass).
# WATCH: tail -f /home/glenn/projects/neuro/logs/lifexact_screen.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1          # 5080 ONLY
export PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/lifexact_screen.log
DONE=logs/lifexact_screen.done.d
mkdir -p "$DONE"
exec 9>logs/lifexact_screen.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[lifexact $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for profiler run to finish"
while pgrep -f '[p]rofile_train.py' >/dev/null; do sleep 30; done
say "=== lif_exact 30M screen START/RESUME ==="
run_stage(){ # $1=run_name $2=config
  local tag=$1 cfg=$2 ec
  if [ -f "$DONE/$tag" ]; then say "[$tag] already complete -- skip"; return 0; fi
  say "[$tag] START/RESUME $cfg"
  "$PY" src/train_gpu.py --config "$cfg" --device cuda >> "$LOG" 2>&1
  ec=$?
  if [ "$ec" -eq 0 ]; then touch "$DONE/$tag"; say "[$tag] COMPLETE"; return 0; fi
  say "[$tag] STOPPED exit=$ec"; return "$ec"
}
run_stage screen_lifexact_s0_30m configs/screen_lifexact_s0_30m.yaml || exit 1
run_stage screen_lifexact_s1_30m configs/screen_lifexact_s1_30m.yaml || exit 1
R=screen_indepwin_s0_30m,screen_indepwin_s1_30m,screen_lifexact_s0_30m,screen_lifexact_s1_30m
if [ ! -f "$DONE/D" ]; then
  say "[D] recall_eval --replay split (B=32)"
  "$PY" scripts/recall_eval.py --runs $R --Ns 4,8,16,32,64 --B 32 --replay split >> "$LOG" 2>&1 && touch "$DONE/D" || say "[D] recall_eval failed"
fi
if [ ! -f "$DONE/E" ]; then
  say "[E] recall_eval single-pass (B=32)"
  "$PY" scripts/recall_eval.py --runs $R --Ns 4,8,16,32,64 --B 32 >> "$LOG" 2>&1 && touch "$DONE/E" || say "[E] recall_eval failed"
fi
say "=== lif_exact 30M screen DONE ==="
