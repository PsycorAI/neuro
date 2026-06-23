#!/usr/bin/env bash
# Fidelity check: train hebb + delta at 30M params / 100M tokens on 5090,
# then run recall_eval. Verdict: does 30M screening reproduce 90M's Hebb>delta?
#   bash scripts/run_fidelity_check.sh > logs_fidelity_check.txt 2>&1 &
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
stamp() { date -Iseconds; }

for cfg in screen_hebb_30m screen_delta_30m; do
  echo ""; echo "===== ${cfg} start $(stamp) ====="
  "$PY" -u src/train_5090_gpu.py --config "configs/${cfg}.yaml" --device cuda \
      2>&1 | tee "logs_${cfg}.txt"
done

echo ""; echo "===== fidelity recall_eval $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs screen_hebb_30m,screen_delta_30m --fpt_K 60
echo "===== done $(stamp) ====="
