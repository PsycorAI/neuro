#!/usr/bin/env bash
# Same as launch_phase25.sh but targets GPU 1 (5080). Use for parallel runs.
#   bash scripts/launch_phase25_5080.sh configs/phase3_5080_tf.yaml
set -e
CFG="${1:-configs/phase3_5080_tf.yaml}"
NAME="$(basename "$CFG" .yaml)"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
{
  echo "===== relaunched $(date -Iseconds) ====="
} >> "logs_${NAME}.txt"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/glenn/projects/neuro/.compile_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
CUDA_VISIBLE_DEVICES=1 nohup "$PY" -u src/train_5080_gpu.py --config "$CFG" --device cuda \
  >> "logs_${NAME}.txt" 2>&1 &
echo "launched $NAME on GPU 1 (5080) (pid $!) -> logs_${NAME}.txt"
echo "watch: tail -f /home/glenn/projects/neuro/logs_${NAME}.txt"
