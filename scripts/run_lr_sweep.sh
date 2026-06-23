#!/usr/bin/env bash
# LR sweep on gated_delta 30M: 5 LR scales x ~25 min each = ~2h total.
# Tests the 2025 paper finding that LR is the biggest overlooked factor for SSM recall.
#   bash scripts/run_lr_sweep.sh > logs_lr_sweep.txt 2>&1 &
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
stamp() { date -Iseconds; }

for cfg in screen_gated_delta_30m_lr0_25 screen_gated_delta_30m_lr0_5 \
           screen_gated_delta_30m screen_gated_delta_30m_lr2_0 \
           screen_gated_delta_30m_lr4_0; do
  echo ""; echo "===== ${cfg} start $(stamp) ====="
  "$PY" -u src/train_5090_gpu.py --config "configs/${cfg}.yaml" --device cuda \
      2>&1 | tee "logs_${cfg}.txt"
done

echo ""; echo "===== LR sweep recall_eval $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs \
  screen_gated_delta_30m_lr0_25,screen_gated_delta_30m_lr0_5,screen_gated_delta_30m,screen_gated_delta_30m_lr2_0,screen_gated_delta_30m_lr4_0,screen_hebb_30m \
  --fpt_K 60
echo "===== done $(stamp) ====="
