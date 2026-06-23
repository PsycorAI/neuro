#!/usr/bin/env bash
# Robust auto-resume wrapper. Restarts the trainer on crash/OOM until the run hits
# its max_tokens (trainer always resumes from latest ckpt, so progress is preserved).
# Sleeps 60s between retries so the GPU can settle.
#   bash scripts/run_until_done.sh 0 configs/gated_delta_90m.yaml > logs_gated_delta_90m.txt 2>&1 &
set -u
GPU="${1:?usage: GPU_INDEX CONFIG}"
CFG="${2:?usage: GPU_INDEX CONFIG}"
NAME="$(basename "$CFG" .yaml)"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
TRAIN=$([ "$GPU" = "0" ] && echo src/train_5090_gpu.py || echo src/train_5080_gpu.py)
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
RETRY=0
while true; do
  echo "===== ${NAME} attempt $((RETRY+1)) $(date -Iseconds) ====="
  "$PY" -u "$TRAIN" --config "$CFG" --device cuda 2>&1 | tee /tmp/last_${NAME}.txt
  ec=${PIPESTATUS[0]}
  if grep -q "^done:" /tmp/last_${NAME}.txt 2>/dev/null; then
    echo "===== ${NAME} COMPLETED (exit $ec) ====="; break
  fi
  RETRY=$((RETRY+1))
  if [ "$RETRY" -ge 20 ]; then echo "===== ${NAME} too many retries; giving up ====="; break; fi
  echo "===== ${NAME} exited $ec; resuming in 60s ====="
  sleep 60
done
