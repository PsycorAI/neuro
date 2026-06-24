#!/usr/bin/env bash
# MDS LR sweep: 4 new LRs + baseline already trained (screen_gated_delta_30m_mqar).
#   bash scripts/run_mds_lr_sweep.sh [GPU] > logs_mds_lr_sweep.txt 2>&1 &
set -e
GPU="${1:-0}"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
TRAIN=$([ "$GPU" = "0" ] && echo src/train_5090_gpu.py || echo src/train_5080_gpu.py)
stamp() { date -Iseconds; }

for cfg in screen_gated_delta_30m_mqar_lr0_25 screen_gated_delta_30m_mqar_lr0_5 \
           screen_gated_delta_30m_mqar_lr2_0 screen_gated_delta_30m_mqar_lr4_0; do
  echo ""; echo "===== ${cfg} start $(stamp) ====="
  "$PY" -u "$TRAIN" --config "configs/${cfg}.yaml" --device cuda 2>&1 | tee "logs_${cfg}.txt"
done

RUNS=screen_gated_delta_30m_mqar,screen_gated_delta_30m_mqar_lr0_25,screen_gated_delta_30m_mqar_lr0_5,screen_gated_delta_30m_mqar_lr2_0,screen_gated_delta_30m_mqar_lr4_0
echo ""; echo "===== MDS-LR recall_eval (single) $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs "$RUNS" --fpt_K 60
echo ""; echo "===== MDS-LR recall_eval (split-replay) $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs "$RUNS" --fpt_K 60 --replay split
echo "===== done $(stamp) ====="
