#!/usr/bin/env bash
# Variants sweep: MQAR-aug (2 arms) + multi-head (3 arms) at 30M screening tier.
# Run AFTER lr sweep finishes. ~25 min/arm * 5 = ~2h.
#   bash scripts/run_variants_sweep.sh > logs_variants_sweep.txt 2>&1 &
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
stamp() { date -Iseconds; }

for cfg in screen_hebb_30m_mqar screen_gated_delta_30m_mqar \
           screen_mh4_gated_delta_30m screen_mh8_gated_delta_30m \
           screen_mh16_gated_delta_30m; do
  echo ""; echo "===== ${cfg} start $(stamp) ====="
  "$PY" -u src/train_5090_gpu.py --config "configs/${cfg}.yaml" --device cuda \
      2>&1 | tee "logs_${cfg}.txt"
done

echo ""; echo "===== variants recall_eval $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs \
  screen_hebb_30m,screen_delta_30m,screen_gated_delta_30m,screen_hebb_30m_mqar,screen_gated_delta_30m_mqar,screen_mh4_gated_delta_30m,screen_mh8_gated_delta_30m,screen_mh16_gated_delta_30m \
  --fpt_K 60
echo "===== done $(stamp) ====="
