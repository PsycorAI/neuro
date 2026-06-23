#!/usr/bin/env bash
# Variants sweep: MQAR-aug + multi-head + pre-conv at 30M screening tier.
# ~25 min/arm * 7 = ~3h on one GPU.
#   bash scripts/run_variants_sweep.sh [GPU_INDEX] > logs_variants_sweep.txt 2>&1 &
set -e
GPU="${1:-1}"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
TRAIN=$([ "$GPU" = "0" ] && echo src/train_5090_gpu.py || echo src/train_5080_gpu.py)
stamp() { date -Iseconds; }

for cfg in screen_hebb_30m_mqar screen_gated_delta_30m_mqar \
           screen_mh4_gated_delta_30m screen_mh8_gated_delta_30m \
           screen_mh16_gated_delta_30m \
           screen_preconv_gated_delta_30m screen_preconv_hebb_30m; do
  echo ""; echo "===== ${cfg} start $(stamp) ====="
  "$PY" -u "$TRAIN" --config "configs/${cfg}.yaml" --device cuda \
      2>&1 | tee "logs_${cfg}.txt"
done

echo ""; echo "===== variants recall_eval $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs \
  screen_hebb_30m,screen_delta_30m,screen_gated_delta_30m,screen_hebb_30m_mqar,screen_gated_delta_30m_mqar,screen_mh4_gated_delta_30m,screen_mh8_gated_delta_30m,screen_mh16_gated_delta_30m,screen_preconv_gated_delta_30m,screen_preconv_hebb_30m \
  --fpt_K 60
echo "===== done $(stamp) ====="
