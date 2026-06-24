#!/usr/bin/env bash
# LR sweep on the RECALL WINNERS (Hebbian, bf05). 4 new LRs each + baselines already
# trained. Tests whether our 2025-paper-informed default LR is actually optimal for
# the winners (not just for the loser gated_delta). ~25 min/arm * 9 new = ~3.75h.
# (screen_hebb_30m and screen_bf05_30m baselines already exist; only new LRs trained.)
#   bash scripts/run_lr_winners_sweep.sh [GPU] > logs_lr_winners_sweep.txt 2>&1 &
set -e
GPU="${1:-0}"
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"
TRAIN=$([ "$GPU" = "0" ] && echo src/train_5090_gpu.py || echo src/train_5080_gpu.py)
stamp() { date -Iseconds; }

# bf05 baseline not yet trained — include it.
for cfg in screen_bf05_30m \
           screen_hebb_30m_lr0_25 screen_hebb_30m_lr0_5 screen_hebb_30m_lr2_0 screen_hebb_30m_lr4_0 \
           screen_bf05_30m_lr0_25 screen_bf05_30m_lr0_5 screen_bf05_30m_lr2_0 screen_bf05_30m_lr4_0; do
  echo ""; echo "===== ${cfg} start $(stamp) ====="
  "$PY" -u "$TRAIN" --config "configs/${cfg}.yaml" --device cuda \
      2>&1 | tee "logs_${cfg}.txt"
done

RUNS=screen_hebb_30m,screen_hebb_30m_lr0_25,screen_hebb_30m_lr0_5,screen_hebb_30m_lr2_0,screen_hebb_30m_lr4_0,screen_bf05_30m,screen_bf05_30m_lr0_25,screen_bf05_30m_lr0_5,screen_bf05_30m_lr2_0,screen_bf05_30m_lr4_0
echo ""; echo "===== winners LR recall_eval (single-pass) $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs "$RUNS" --fpt_K 60
echo ""; echo "===== winners LR recall_eval (split-replay) $(stamp) ====="
"$PY" -u scripts/recall_eval.py --runs "$RUNS" --fpt_K 60 --replay split
echo "===== done $(stamp) ====="
