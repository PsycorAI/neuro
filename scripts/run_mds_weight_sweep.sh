#!/usr/bin/env bash
# MDS MQAR-weight sweep: weights 0.25, 1.0, 2.0 + baseline 0.5 already trained.
#   bash scripts/run_mds_weight_sweep.sh [GPU] > logs_mds_weight_sweep.txt 2>&1 &
set -e
GPU="${1:-1}"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
TRAIN=$([ "$GPU" = "0" ] && echo src/train_5090_gpu.py || echo src/train_5080_gpu.py)
stamp() { date -Iseconds; }

for cfg in screen_gated_delta_30m_mqar_w0_25 screen_gated_delta_30m_mqar_w1_0 \
           screen_gated_delta_30m_mqar_w2_0; do
  echo ""; echo "===== ${cfg} start $(stamp) ====="
  "$PY" -u "$TRAIN" --config "configs/${cfg}.yaml" --device cuda 2>&1 | tee "logs_${cfg}.txt"
done

RUNS=screen_gated_delta_30m_mqar,screen_gated_delta_30m_mqar_w0_25,screen_gated_delta_30m_mqar_w1_0,screen_gated_delta_30m_mqar_w2_0
echo ""; echo "===== MDS-weight recall_eval (single) $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs "$RUNS" --fpt_K 60
echo ""; echo "===== MDS-weight recall_eval (split-replay) $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs "$RUNS" --fpt_K 60 --replay split
echo "===== done $(stamp) ====="
