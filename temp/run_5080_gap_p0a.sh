#!/bin/bash
# run_5080_gap_p0a.sh -- 2026-09-27 ppl-gap plan, first three steps (docs/ppl_gap_plan_2026_09_27.md), Glenn-approved.
#   0. recall for lif_exact s1 (the honest runner was stopped before its recall step; FPT s1 skipped)
#   P0. transformer AdamW LR check: 5e-4, 2e-3 (control 1e-3 = 68.5)
#   A1. spiking NorMuon 0.02 / AdamW 1e-3     A2. spiking AdamW 1e-3     (control lif_exact s0/s1 = 99.3 / 106.6)
#   A3. 2-layer [spk, attn] hybrid with the A1/A2 winner (temp/gap_pick_hybrid.py)
# Auto-kill (spiking arms): eval2 > 170 (control eval2 mean 154 x 1.10). Recall = single-pass MQAR N4..64.
# WATCH: tail -f /home/glenn/projects/neuro/logs/gap_screen.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_screen.log
exec 9>logs/gap_screen.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[gap $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
recall(){ "$PY" scripts/recall_eval.py --runs "$1" --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E '^ *(run|screen|chance)' >> "$LOG"; }
train(){ # $1 = run name, $2 = kill threshold for eval2 (0 = no kill)
  local run=$1 alog=logs/$1.log pid n v
  rm -rf models/checkpoints/$run
  say "[$run] START"
  "$PY" src/train_gpu.py --config configs/$run.yaml --device cuda > "$alog" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    n=$(grep -c 'eval\] val_ppl' "$alog")
    v=$(grep 'eval\] val_ppl' "$alog" | tail -1 | awk '{print $3}')
    if [ "$2" != "0" ] && [ "$n" -eq 2 ] && awk "BEGIN{exit !($v > $2)}"; then
      say "[$run] KILL eval2 $v > $2"; kill -INT $pid; wait $pid; return 1
    fi
  done
  wait $pid; say "[$run] exit=$? :: $(grep 'eval\] val_ppl' "$alog" | awk '{print $3}' | tr '\n' ' ')"
}
say "=== ppl-gap P0/A1/A2/A3 queue START ==="
recall screen_honest_lifexact_s0_30m,screen_honest_lifexact_s1_30m,screen_honest_fpt_s0_30m
train screen_tfmr_adamw_lr5e4_s0_30m 0
train screen_tfmr_adamw_lr2e3_s0_30m 0
train screen_honest_mu02_s0_30m 170 && recall screen_honest_mu02_s0_30m
train screen_honest_adamw_s0_30m 170 && recall screen_honest_adamw_s0_30m
/usr/bin/python3 temp/gap_pick_hybrid.py 2>&1 | tee -a "$LOG"
train screen_honest_hybrid_s0_30m 170 && recall screen_honest_hybrid_s0_30m
say "=== DONE ==="
