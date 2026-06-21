#!/usr/bin/env bash
# ST Phase 1.5 — does long-lived memory (lam=0.995) unlock the stateful benefit?
# Trains the stateful-lam995 arm, then runs the 2x2 stateful eval across all three
# runs (random baseline, stateful-lam098, stateful-lam995).
#   nohup bash scripts/run_st_ab2.sh > logs_st_ab2.txt 2>&1 &
#   tail -f logs_st_ab2.txt
set -e
cd /home/glenn/projects/neuro
PY=/home/glenn/projects/bdh/venv/bin/python
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/home/glenn/projects/neuro/.compile_cache
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
stamp() { date -Iseconds; }

echo "[ST Phase 1.5] started $(stamp)"

echo ""
echo "===== stateful + lam=0.995 ($(stamp)) ====="
rm -rf models/checkpoints/phase25_st_stateful_lam995
"$PY" -u src/train_5080_gpu.py --config configs/phase25_st_stateful_lam995.yaml \
    --device cuda 2>&1 | tee logs_st_stateful_lam995.txt

echo ""
echo "===== 2x2 stateful eval across all three runs ($(stamp)) ====="
# Note: stateful_eval builds each model from its own saved cfg, so lam is honored.
"$PY" -u scripts/stateful_eval.py \
    --runs phase25_st_random,phase25_st_stateful,phase25_st_stateful_lam995 \
    --eval_tokens 300000 2>&1 | tee logs_st_eval_3way.txt

echo ""
echo "[ST Phase 1.5] done $(stamp)"
