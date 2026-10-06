#!/bin/bash
# run_5080_honest.sh -- 2026-09-27: first HONESTLY-trained 30M runs (leak-fixed lif_parallel, commit 2e98665).
# FPT K=10 control vs lif_exact (exact LIF kernel), same config/seed/100M tok. Compare to tfmr AdamW 68.5.
# WATCH: tail -f /home/glenn/projects/neuro/logs/honest_screen.log
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/honest_screen.log
exec 9>logs/honest_screen.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[honest $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
for r in screen_honest_lifexact_s1_30m screen_honest_fpt_s1_30m; do
  rm -rf models/checkpoints/$r
  say "[$r] START"
  $PY src/train_gpu.py --config configs/$r.yaml --device cuda > logs/$r.log 2>&1
  say "[$r] exit=$? :: $(grep 'eval\]' logs/$r.log | awk '{print $3}' | tr '\n' ' ')"
done
$PY scripts/recall_eval.py --runs screen_honest_fpt_s0_30m,screen_honest_fpt_s1_30m,screen_honest_lifexact_s0_30m,screen_honest_lifexact_s1_30m --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E '^ *(run|screen|chance)' >> "$LOG"
say "DONE"
