#!/bin/bash
# run_5080_gap_hyb.sh -- 2026-09-28 morning, 5080 free until ~13:00 (Glenn). Hybrid [spk, attn] = 82.4 (vs lif_exact 99.3/106.6).
#   1. position diagnostic on lif_exact vs hybrid vs learnbeta (vs tfmr)  2. hybrid seed 1  3. hybrid + lif_learn_beta (s0)
# No new run starts after 10:15 (must be free by 13:00). WATCH: tail -f /home/glenn/projects/neuro/logs/gap_screen.log
set -u
cd /home/glenn/projects/neuro
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
PY=/home/glenn/projects/bdh/venv/bin/python
LOG=logs/gap_screen.log
exec 9>logs/gap_screen.lock; flock -n 9 || { echo "already running"; exit 1; }
say(){ echo "[gap $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
CK=/home/glenn/projects/neuro/models/checkpoints
"$PY" temp/diag_gap_positions.py lif_exact=$CK/screen_honest_lifexact_s0_30m/step_0006104.pt hybrid=$CK/screen_honest_hybrid_s0_30m/step_0006104.pt learnbeta=$CK/screen_gap_learnbeta_s0_30m/step_0006104.pt > logs/diag_gap_positions_hybrid.log 2>&1
say "diag done -> logs/diag_gap_positions_hybrid.log"
for run in screen_honest_hybrid_s1_30m screen_gap_hyb_learnbeta_s0_30m; do
  if [ "$(date +%H%M)" -gt 1015 ]; then say "[$run] SKIPPED (past 10:15, 5080 needed at 13:00)"; continue; fi
  rm -rf models/checkpoints/$run
  say "[$run] START"
  "$PY" src/train_gpu.py --config configs/$run.yaml --device cuda > logs/$run.log 2>&1
  say "[$run] exit=$? :: $(grep 'eval\] val_ppl' logs/$run.log | awk '{print $3}' | tr '\n' ' ')"
  "$PY" scripts/recall_eval.py --runs $run --Ns 4,8,16,32,64 --B 32 2>&1 | grep -E "^ *$run" | head -1 >> "$LOG"
done
say "=== 5080 hybrid follow-ups DONE ==="
