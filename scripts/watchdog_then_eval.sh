#!/usr/bin/env bash
# Wait for the running phase3_5090 training to finish, then auto-run eval.
# Launch with:
#   nohup bash scripts/watchdog_then_eval.sh > logs_watchdog.txt 2>&1 &

set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
LOG="logs_watchdog.txt"

echo "[watchdog] started at $(date -Iseconds)" | tee -a "$LOG"

# Wait until no train_gpu.py process exists (training exited).
while pgrep -f "train_gpu.py" > /dev/null; do
    sleep 60
done

echo "[watchdog] training process exited at $(date -Iseconds)" | tee -a "$LOG"

# Give the OS a moment to flush the final checkpoint.
sleep 10

# Run eval on the latest checkpoint of phase3_5090.
echo "[watchdog] launching eval_phase3.py" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_phase3.py --iters 100 >> "$LOG" 2>&1 || \
    echo "[watchdog] eval_phase3 failed" | tee -a "$LOG"

echo "[watchdog] done at $(date -Iseconds)" | tee -a "$LOG"
