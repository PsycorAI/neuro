#!/usr/bin/env bash
# Launch a Phase-2.5 run on the 5080 (GPU 0), detached, logging to logs_<run>.txt.
#   bash scripts/launch_phase25.sh configs/phase25_a.yaml
# Resumes automatically from the latest checkpoint if present.
set -e
CFG="${1:-configs/phase25_a.yaml}"
NAME="$(basename "$CFG" .yaml)"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
# Append (>>) so a relaunch / resume preserves the historical eval timeline
# instead of truncating it. A separator line marks each restart.
{
  echo "===== relaunched $(date -Iseconds) ====="
} >> "logs_${NAME}.txt"
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u src/train_gpu.py --config "$CFG" --device cuda \
  >> "logs_${NAME}.txt" 2>&1 &
echo "launched $NAME (pid $!) -> logs_${NAME}.txt"
echo "watch: tail -f /home/glenn/projects/neuro/logs_${NAME}.txt"
